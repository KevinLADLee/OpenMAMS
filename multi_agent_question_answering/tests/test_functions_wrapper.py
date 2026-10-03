import json

import pytest
from langchain_core.messages import AIMessage

from remembr.tools.functions_wrapper import FunctionsWrapper


class FixedModel:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        return AIMessage(content=self.content)


def test_repeated_calls_do_not_grow_tool_definitions():
    functions = [{"name": "retrieve_from_text", "parameters": {"type": "object"}}]
    model = FixedModel(json.dumps({"tool": "retrieve_from_text", "tool_input": {"x": "bus"}}))
    wrapper = FunctionsWrapper(model)
    for _ in range(3):
        message = wrapper._generate([], functions=functions).generations[0].message
        assert message.tool_calls[0]["args"] == {"x": "bus"}
    assert functions == [{"name": "retrieve_from_text", "parameters": {"type": "object"}}]
    assert len(set(model.prompts)) == 1


def test_missing_tool_is_reported_with_model_output():
    content = '{"response":"A bus was observed."}'
    with pytest.raises(ValueError, match="tool") as exc:
        FunctionsWrapper(FixedModel(content))._generate([])
    assert content in str(exc.value)


def test_missing_conversation_response_never_enters_debugger(monkeypatch):
    def forbidden_debugger():
        raise AssertionError("Production inference must not enter pdb")
    monkeypatch.setattr("pdb.set_trace", forbidden_debugger)
    with pytest.raises(ValueError, match="response"):
        FunctionsWrapper(FixedModel('{"tool":"__conversational_response"}'))._generate([])


def test_malformed_model_response_is_corrected_with_feedback():
    class CorrectingModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return AIMessage(content='[{"tool_reasoning":"Search for a black car."}]')
            assert 'tool' in messages[-1].content
            assert messages[-2].content == '[{"tool_reasoning":"Search for a black car."}]'
            return AIMessage(content='{"tool":"retrieve_from_text","tool_input":{"x":"black car"}}')

    model = CorrectingModel()
    result = FunctionsWrapper(model)._generate(
        [], functions=[{"name": "retrieve_from_text", "parameters": {"type": "object"}}]
    )
    assert len(model.calls) == 2
    assert result.generations[0].message.tool_calls[0]['args'] == {'x': 'black car'}


def test_format_correction_attempts_are_bounded():
    model = FixedModel('{"tool_reasoning":"Missing tool."}')
    with pytest.raises(ValueError, match='tool'):
        FunctionsWrapper(model)._generate([])
    assert len(model.prompts) == 2
