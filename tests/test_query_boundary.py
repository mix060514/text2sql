import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from text2sql.agents.planned_agent import tools


class QueryBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "sales.sqlite"
        with sqlite3.connect(self.database) as connection:
            connection.execute('CREATE TABLE sales_data ("Order ID" TEXT, Region TEXT, Quantity INTEGER, "Total Revenue" REAL)')
            connection.executemany('INSERT INTO sales_data VALUES (?, ?, ?, ?)', [
                ("A", "APAC", 2, 10.0), ("B", "APAC", 4, 20.0), ("C", "EMEA", 1, 7.0),
            ])
            connection.execute('CREATE TABLE private_data (secret TEXT)')
            connection.execute("INSERT INTO private_data VALUES ('private')")

    def tearDown(self):
        self.temp.cleanup()

    def query(self, sql):
        return tools.read_sales(sql, self.database)

    def test_real_sqlite_syntax_validation(self):
        with self.assertRaises(tools.QueryError):
            self.query('SELECT definitely_not_a_column FROM sales_data')

    def test_count_star_and_distinct_orders(self):
        self.assertEqual(self.query('SELECT COUNT(*) AS n FROM sales_data')["rows"], [[3]])
        self.assertEqual(self.query('SELECT COUNT(DISTINCT "Order ID") AS n FROM sales_data')["rows"], [[3]])

    def test_cte_and_window_function(self):
        result = self.query('WITH counts AS (SELECT Region, SUM(Quantity) AS n FROM sales_data GROUP BY Region) SELECT Region, n, ROW_NUMBER() OVER (ORDER BY n DESC) AS rank FROM counts ORDER BY rank')
        self.assertEqual(result["rows"], [["APAC", 6, 1], ["EMEA", 1, 2]])

    def test_unsafe_queries_cannot_change_database(self):
        before = hashlib.sha256(self.database.read_bytes()).hexdigest()
        queries = [
            'DELETE FROM sales_data', 'DROP TABLE sales_data',
            'UPDATE sales_data SET Quantity = 0', 'PRAGMA writable_schema = ON',
            "ATTACH DATABASE ':memory:' AS other", 'SELECT * FROM sales_data; DELETE FROM sales_data',
            "SELECT load_extension('anything') FROM sales_data",
            'WITH x AS (SELECT * FROM sales_data) DELETE FROM sales_data',
        ]
        for sql in queries:
            with self.subTest(sql=sql), self.assertRaises(tools.QueryError):
                self.query(sql)
        self.assertEqual(hashlib.sha256(self.database.read_bytes()).hexdigest(), before)

    def test_other_tables_and_metadata_are_denied(self):
        for sql in ('SELECT * FROM private_data', 'SELECT * FROM sqlite_master',
                    'SELECT * FROM sales_data, private_data'):
            with self.subTest(sql=sql), self.assertRaises(tools.QueryError):
                self.query(sql)

    def test_constants_do_not_count_as_sales_evidence(self):
        with self.assertRaises(tools.QueryError) as caught:
            self.query('SELECT 123 AS Revenue')
        self.assertEqual(caught.exception.code, "NO_SALES_DATA")

    def test_empty_result_is_success(self):
        result = self.query("SELECT Region FROM sales_data WHERE Region = 'missing'")
        self.assertTrue(result["success"])
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["columns"], ["Region"])

    def test_no_silent_row_truncation(self):
        with patch.object(tools, "MAX_QUERY_ROWS", 1), self.assertRaises(tools.QueryError) as caught:
            self.query('SELECT * FROM sales_data')
        self.assertEqual(caught.exception.code, "ROW_LIMIT")

    def test_result_size_is_bounded(self):
        with patch.object(tools, "MAX_RESULT_BYTES", 20), self.assertRaises(tools.QueryError) as caught:
            self.query('SELECT * FROM sales_data')
        self.assertEqual(caught.exception.code, "RESULT_SIZE")

    def test_duplicate_aliases_rejected(self):
        with self.assertRaises(tools.QueryError) as caught:
            self.query('SELECT Region AS x, Quantity AS x FROM sales_data')
        self.assertEqual(caught.exception.code, "DUPLICATE_COLUMNS")

    def test_failed_attempt_invalidates_success_and_chart(self):
        state = {"query_success": True, "raw_query_result": [[999]],
                 "column_names": ["old"], "chart_json": "old", "final_answer": "old"}
        result = tools.execute_query('SELECT missing FROM sales_data', state)
        self.assertFalse(result["success"])
        self.assertFalse(state["query_success"])
        self.assertEqual(state["raw_query_result"], [])
        self.assertEqual(state["column_names"], [])
        self.assertIsNone(state["chart_json"])
        self.assertEqual(state["final_answer"], "")

    def test_query_time_limit(self):
        with sqlite3.connect(self.database) as connection:
            connection.executemany('INSERT INTO sales_data VALUES (?, ?, ?, ?)',
                                   [(str(i), "APAC", 1, 1.0) for i in range(100)])
        with patch.object(tools, "QUERY_TIMEOUT_SECONDS", -1), self.assertRaises(tools.QueryError) as caught:
            self.query('SELECT SUM(a.Quantity * b.Quantity * c.Quantity) FROM sales_data a, sales_data b, sales_data c')
        self.assertEqual(caught.exception.code, "QUERY_TIMEOUT")


if __name__ == "__main__":
    unittest.main()
