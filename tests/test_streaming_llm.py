import asyncio
import unittest

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from text2sql.agents.planned_agent.qwen_workflow import BoundedStreamingLiteLlm


def chunk(text=None, finish=None):
    return {"choices": [{"delta": {"content": text} if text else {"role": "assistant"},
                         "finish_reason": finish}]}


class FakeStream:
    def __init__(self, chunks, delay=0.0):
        self.chunks, self.delay, self.closed, self.kwargs = list(chunks), delay, False, None

    async def acompletion(self, **kwargs):
        self.kwargs = kwargs
        return self._gen()

    async def _gen(self):
        try:
            for item in self.chunks:
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield item
        finally:
            self.closed = True  # the HTTP stream is closed -> llama.cpp cancels the slot


def request():
    return LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="q")])])


async def collect(model):
    return [r async for r in model.generate_content_async(request(), stream=False)]


class BoundedStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_but_yields_only_the_final_response(self):
        model = BoundedStreamingLiteLlm(model="openai/x", request_seconds=5)
        model.llm_client = fake = FakeStream([chunk('{"sql": '), chunk('"SELECT 1"}'), chunk(finish="stop")])
        responses = await collect(model)
        self.assertTrue(fake.kwargs["stream"])
        self.assertEqual(len(responses), 1)
        self.assertFalse(responses[0].partial)
        self.assertEqual(responses[0].content.parts[0].text, '{"sql": "SELECT 1"}')

    async def test_truncated_stream_still_returns_its_text(self):
        model = BoundedStreamingLiteLlm(model="openai/x", request_seconds=5)
        model.llm_client = FakeStream([chunk('{"sql": "SEL'), chunk(finish="length")])
        responses = await collect(model)
        self.assertEqual([r.content.parts[0].text for r in responses], ['{"sql": "SEL'])

    async def test_total_deadline_closes_the_stream(self):
        model = BoundedStreamingLiteLlm(model="openai/x", request_seconds=0.2)
        model.llm_client = fake = FakeStream([chunk("a")] * 100, delay=0.05)
        with self.assertRaises(TimeoutError):
            await collect(model)
        self.assertTrue(fake.closed)

    async def test_no_per_request_limit_by_default(self):
        # The only time limit is the invocation deadline (runtime.stream_turn).
        from text2sql.agents.planned_agent import qwen_workflow as qw

        self.assertIsNone(BoundedStreamingLiteLlm(model="openai/x").request_seconds)
        self.assertIsNone(qw.agent_sql.model.request_seconds)
        model = BoundedStreamingLiteLlm(model="openai/x")
        model.llm_client = FakeStream([chunk("a")] * 5 + [chunk(finish="stop")], delay=0.01)
        self.assertEqual((await collect(model))[0].content.parts[0].text, "aaaaa")

    async def test_invocation_deadline_closes_the_stream(self):
        # When the outer turn deadline cancels the request, the stream is still closed,
        # so llama.cpp frees the slot without a per-request timeout.
        model = BoundedStreamingLiteLlm(model="openai/x")
        model.llm_client = fake = FakeStream([chunk("a")] * 100, delay=0.05)
        with self.assertRaises(TimeoutError):
            async with asyncio.timeout(0.2):
                await collect(model)
        self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main()
