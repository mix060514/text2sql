import json
import unittest

from pydantic import ValidationError
from text2sql.agents.planned_agent.contracts import AnswerSubmission
from text2sql.agents.planned_agent.tools import clear_query_state, complete_answer


class PublicationTests(unittest.TestCase):
    def state(self, columns=None, rows=None):
        return {"query_success": True, "column_names": columns or ["Region", "Revenue"],
                "raw_query_result": rows if rows is not None else [["EMEA", 20], ["APAC", 10]]}

    def test_successful_query_required(self):
        with self.assertRaises(ValueError):
            complete_answer(AnswerSubmission(answer="unsupported"), {"query_success": False})

    def test_chart_fields_form_a_contract(self):
        for data in ({"chart_type": "bar"}, {"chart_type": "unsupported"},
                     {"chart_type": "none", "x_column": "Region"},
                     {"chart_type": "pie", "x_column": "Region", "y_column": "Revenue", "color_column": "Region"}):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                AnswerSubmission(answer="answer", **data)

    def test_render_called_directly_and_payload_stays_out_of_answer(self):
        state = self.state()
        result = complete_answer(AnswerSubmission(answer="EMEA 20；APAC 10。", chart_type="bar",
                                                 x_column="Region", y_column="Revenue"), state)
        self.assertTrue(result["success"])
        self.assertNotIn("chart_json", result)
        self.assertEqual(state["final_answer"], "EMEA 20；APAC 10。")
        chart = json.loads(state["chart_json"])
        self.assertEqual(chart["data"][0]["x"], ["EMEA", "APAC"])

    def test_multirow_measure_requires_chart(self):
        with self.assertRaises(ValueError):
            complete_answer(AnswerSubmission(answer="answer"), self.state())

    def test_unknown_columns_rejected(self):
        with self.assertRaises(ValueError):
            complete_answer(AnswerSubmission(answer="answer", chart_type="bar", x_column="made_up", y_column="Revenue"), self.state())

    def test_numeric_y_column_required(self):
        with self.assertRaises(ValueError):
            complete_answer(AnswerSubmission(answer="answer", chart_type="bar", x_column="Revenue", y_column="Region"), self.state())

    def test_empty_result_can_be_published(self):
        state = self.state(rows=[])
        complete_answer(AnswerSubmission(answer="沒有符合條件的資料。"), state)
        self.assertEqual(state["workflow_status"], "completed")
        self.assertIsNone(state["chart_json"])

    def test_numeric_years_are_plain_ordered_labels(self):
        state = self.state(["Year", "Revenue"], [[2023, 10], [2024, 20]])
        complete_answer(AnswerSubmission(answer="2023: 10；2024: 20。", chart_type="bar", x_column="Year", y_column="Revenue"), state)
        self.assertEqual(json.loads(state["chart_json"])["data"][0]["x"], ["2023", "2024"])

    def test_pie_negative_values_rejected(self):
        with self.assertRaises(ValueError):
            complete_answer(AnswerSubmission(answer="answer", chart_type="pie", x_column="Region", y_column="Revenue"), self.state(rows=[["A", -1], ["B", 2]]))

    def test_turn_reset_removes_stale_chart_and_answer(self):
        state = self.state()
        state.update({"chart_json": "old", "answer": "old", "final_answer": "old"})
        clear_query_state(state)
        self.assertIsNone(state["chart_json"])
        self.assertEqual(state["answer"], "")
        self.assertEqual(state["final_answer"], "")
        self.assertFalse(state["query_success"])


if __name__ == "__main__":
    unittest.main()
