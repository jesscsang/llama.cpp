#pragma once

#include "common.h"
#include "log.h"
#include "llama.h"

#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif
// increase max payload length to allow use of larger context size
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 1048576
#include "httplib.h"

// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"

#include <algorithm>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define DEFAULT_OAICOMPAT_MODEL "gpt-3.5-turbo-0613"

using json = nlohmann::ordered_json;

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
};

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const &) {
            LOG_WRN("Wrong type supplied for parameter '%s'. Expected '%s', using default value\n", key.c_str(), json(default_value).type_name());
            return default_value;
        }
    } else {
        return default_value;
    }
}

//
// chat template utils
//

static std::string llama_get_chat_template(const struct llama_model * model) {
    const char * template_key = "tokenizer.chat_template";
    int32_t res = llama_model_meta_val_str(model, template_key, nullptr, 0);
    if (res < 0) {
        return "";
    }

    std::vector<char> model_template(res + 1, 0);
    res = llama_model_meta_val_str(model, template_key, model_template.data(), model_template.size());
    if (res < 0) {
        return "";
    }
    return std::string(model_template.data(), res);
}

inline common_chat_params format_chat(
        const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<json> & messages,
        const json & tools = json(),
        common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO,
        bool parallel_tool_calls = true,
        const std::string & json_schema = "",
        const std::string & grammar = "") {
    auto tmpls = common_chat_templates_init(model, tmpl);

    common_chat_templates_inputs inputs;
    inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
    inputs.tools                 = common_chat_tools_parse_oaicompat(tools);
    inputs.tool_choice           = tool_choice;
    inputs.parallel_tool_calls   = parallel_tool_calls;
    inputs.add_generation_prompt = true;
    inputs.use_jinja             = true;
    inputs.json_schema           = json_schema;
    inputs.grammar               = grammar;

    common_chat_params params = common_chat_templates_apply(tmpls.get(), inputs);
    LOG_DBG("formatted_chat: '%s'\n", params.prompt.c_str());
    return params;
}

//
// base64 utils (TODO: move to common in the future)
//

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(uint8_t c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline std::vector<uint8_t> base64_decode(const std::string & encoded_string) {
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    std::vector<uint8_t> ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }

            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}

//
// random string / id
//

static std::string random_string() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

static std::string gen_chatcmplid() {
    return "chatcmpl-" + random_string();
}

//
// other common utils
//

static size_t longest_common_prefix(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {}

    return i;
}

static size_t longest_common_prefix(const std::string & a, const std::string & b) {
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {}

    return i;
}

static bool ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string &stop, const std::string &text) {
    if (!text.empty() && !stop.empty()) {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--) {
            if (stop[char_index] == text_last_char) {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial)) {
                    return text.size() - char_index - 1;
                }
            }
        }
    }

    return std::string::npos;
}

static bool json_is_array_of_numbers(const json & data) {
    if (data.is_array()) {
        for (const auto & e : data) {
            if (!e.is_number()) {
                return false;
            }
        }
        return true;
    }
    return false;
}

// TODO: reuse llama_detokenize
template <class Iter>
static std::string tokens_to_str(llama_context * ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += common_token_to_piece(ctx, *begin);
    }

    return ret;
}

// format incomplete utf-8 multibyte character for output
static std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token) {
    std::string out = token == -1 ? "" : common_token_to_piece(ctx, token);

    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80) {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }

    return out;
}

struct completion_token_output {
    llama_token tok;
    std::string text_to_send;

    struct token_prob {
        llama_token tok;
        float prob;
    };

    std::vector<token_prob> probs;
};

// convert a vector of completion_token_output to json
static json probs_vector_to_json(const llama_context * ctx, const std::vector<completion_token_output> & probs) {
    json out = json::array();

    for (const auto & prob : probs) {
        json probs_for_token = json::array();

        for (const auto & p : prob.probs) {
            const std::string tok_str = tokens_to_output_formatted_string(ctx, p.tok);
            probs_for_token.push_back(json {
                {"tok_str", tok_str},
                {"prob",    p.prob},
            });
        }

        const std::string tok_str = tokens_to_output_formatted_string(ctx, prob.tok);
        out.push_back(json {
            {"content", tok_str},
            {"probs",   probs_for_token},
        });
    }

    return out;
}

static bool server_sent_event(httplib::DataSink & sink, const char * event, const json & data) {
    const std::string str =
        std::string(event) + ": " +
        data.dump(-1, ' ', false, json::error_handler_t::replace) +
        "\n\n"; // note: these newlines are important (not sure why though, if you know, add a comment to explain)

    LOG_DBG("data stream, to_send: %s", str.c_str());

    return sink.write(str.c_str(), str.size());
}

static std::string gen_tool_call_id() {
    return "call_" + random_string();
}

static std::vector<std::string> tool_names_from_request(const json & tools, const std::string & forced_name = "") {
    std::vector<std::string> names;

    if (!tools.is_array()) {
        return names;
    }

    for (const auto & tool : tools) {
        if (!tool.is_object() || json_value(tool, "type", std::string()) != "function" || !tool.contains("function")) {
            continue;
        }
        const auto & function = tool["function"];
        const std::string name = json_value(function, "name", std::string());
        if (name.empty()) {
            continue;
        }
        if (forced_name.empty() || forced_name == name) {
            names.push_back(name);
        }
    }

    return names;
}

struct parsed_tool_call_content {
    bool has_tool_calls = false;
    std::string content;
    json tool_calls = json::array();
};

static bool tool_name_allowed(const std::string & name, const std::vector<std::string> & allowed_names) {
    if (allowed_names.empty()) {
        return true;
    }

    return std::find(allowed_names.begin(), allowed_names.end(), name) != allowed_names.end();
}

static bool append_tool_call_oaicompat(
        json & out,
        const json & raw_tool_call,
        const std::vector<std::string> & allowed_names = {}) {
    std::string name;
    json arguments = json::object();

    if (raw_tool_call.contains("function")) {
        const auto & function = raw_tool_call["function"];
        name = json_value(function, "name", std::string());
        if (function.contains("arguments")) {
            arguments = function["arguments"];
        }
    } else {
        name = json_value(raw_tool_call, "name", std::string());
        if (raw_tool_call.contains("arguments")) {
            arguments = raw_tool_call["arguments"];
        }
    }

    if (name.empty()) {
        return false;
    }
    if (!tool_name_allowed(name, allowed_names)) {
        return false;
    }

    std::string arguments_str;
    if (arguments.is_string()) {
        arguments_str = arguments.get<std::string>();
    } else {
        arguments_str = arguments.dump(-1, ' ', false, json::error_handler_t::replace);
    }

    json tool_call = {
        {"id",   json_value(raw_tool_call, "id", gen_tool_call_id())},
        {"type", "function"},
        {"function", {
            {"name",      name},
            {"arguments", arguments_str},
        }},
    };

    out.push_back(tool_call);
    return true;
}

static bool parse_tool_calls_json(
        json & out,
        const std::string & raw_json,
        const std::vector<std::string> & allowed_names = {}) {
    json parsed = json::parse(raw_json);
    json raw_calls = json::array();

    if (parsed.is_array()) {
        raw_calls = parsed;
    } else if (parsed.is_object() && parsed.contains("tool_calls") && parsed["tool_calls"].is_array()) {
        raw_calls = parsed["tool_calls"];
    } else if (parsed.is_object()) {
        raw_calls.push_back(parsed);
    }

    for (const auto & raw_tool_call : raw_calls) {
        append_tool_call_oaicompat(out, raw_tool_call, allowed_names);
    }

    return !out.empty();
}

static parsed_tool_call_content parse_tool_calls_from_content(
        const std::string & content,
        const std::vector<std::string> & allowed_names = {},
        bool allow_bare_json = false) {
    static const std::string start_tag = "<tool_call>";
    static const std::string end_tag   = "</tool_call>";

    parsed_tool_call_content result;
    result.content = content;

    const size_t start = content.find(start_tag);
    if (start == std::string::npos) {
        if (allow_bare_json) {
            try {
                const std::string raw_json = string_strip(content);
                if (!raw_json.empty() && parse_tool_calls_json(result.tool_calls, raw_json, allowed_names)) {
                    result.content = "";
                    result.has_tool_calls = true;
                }
            } catch (const std::exception &) {
                // Normal assistant replies can be arbitrary text; only tagged tool
                // calls are noisy enough to log parse failures.
            }
        }
        return result;
    }

    const size_t body_start = start + start_tag.size();
    const size_t end = content.find(end_tag, body_start);
    if (end == std::string::npos) {
        return result;
    }

    const std::string raw_json = string_strip(content.substr(body_start, end - body_start));
    if (raw_json.empty()) {
        return result;
    }

    try {
        parse_tool_calls_json(result.tool_calls, raw_json, allowed_names);
    } catch (const std::exception & e) {
        LOG_WRN("failed to parse tool calls: %s\n", e.what());
        return result;
    }

    if (!result.tool_calls.empty()) {
        const std::string before = content.substr(0, start);
        const std::string after  = content.substr(end + end_tag.size());
        result.content = string_strip(before + after);
        result.has_tool_calls = true;
    }

    return result;
}

static parsed_tool_call_content parse_tool_calls_for_request(const std::string & content, const json & request) {
    const bool request_has_tools =
        request.contains("tools") && request["tools"].is_array() && !request["tools"].empty() &&
        json_value(request, "tool_choice", std::string("auto")) != "none";
    const std::vector<std::string> allowed_names =
        request_has_tools ? tool_names_from_request(request["tools"]) : std::vector<std::string>();

    if (request_has_tools && request.contains("chat_parser") && request["chat_parser"].is_string()) {
        try {
            common_chat_parser_params parser_params;
            parser_params.format = static_cast<common_chat_format>(json_value(request, "chat_format", (int) COMMON_CHAT_FORMAT_CONTENT_ONLY));
            parser_params.generation_prompt = json_value(request, "generation_prompt", std::string());
            parser_params.parse_tool_calls = true;
            parser_params.parser.load(request["chat_parser"].get<std::string>());

            common_chat_msg parsed_msg = common_chat_parse(content, false, parser_params);
            parsed_tool_call_content parsed;
            parsed.content = parsed_msg.render_content();

            for (const auto & tool_call : parsed_msg.tool_calls) {
                append_tool_call_oaicompat(parsed.tool_calls, {
                    {"id", tool_call.id},
                    {"function", {
                        {"name", tool_call.name},
                        {"arguments", tool_call.arguments},
                    }},
                }, allowed_names);
            }

            if (!parsed.tool_calls.empty()) {
                parsed.has_tool_calls = true;
                parsed.content = string_strip(parsed.content);
                return parsed;
            }
        } catch (const std::exception & e) {
            LOG_DBG("failed to parse tool calls with chat parser: %s\n", e.what());
        }
    }

    return parse_tool_calls_from_content(content, allowed_names, request_has_tools);
}

//
// OAI utils
//

static json oaicompat_completion_params_parse(
    const struct llama_model * model,
    const json & body, /* openai api json semantics */
    const std::string & chat_template) {
    json llama_params;

    llama_params["__oaicompat"] = true;

    json tools_for_prompt;
    bool tools_enabled = false;
    common_chat_tool_choice chat_tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    bool parallel_tool_calls = json_value(body, "parallel_tool_calls", true);
    std::string forced_tool_name;
    json json_schema;
    std::string grammar = json_value(body, "grammar", std::string());

    if (body.contains("tools") && !body.at("tools").is_null()) {
        if (!body.at("tools").is_array()) {
            throw std::runtime_error("tools must be an array");
        }
        if (!body.at("tools").empty()) {
            tools_enabled = true;
            tools_for_prompt = body.at("tools");
        }
    }

    if (body.contains("tool_choice") && !body.at("tool_choice").is_null()) {
        const json & tool_choice = body.at("tool_choice");
        if (tool_choice.is_string()) {
            const std::string choice = tool_choice.get<std::string>();
            if (choice == "none") {
                tools_enabled = false;
                tools_for_prompt = json();
                chat_tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
            } else if (choice == "required") {
                chat_tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            } else {
                chat_tool_choice = common_chat_tool_choice_parse_oaicompat(choice);
            }
        } else if (tool_choice.is_object()) {
            if (json_value(tool_choice, "type", std::string()) != "function" || !tool_choice.contains("function")) {
                throw std::runtime_error("tool_choice object must be of type \"function\"");
            }
            forced_tool_name = json_value(tool_choice["function"], "name", std::string());
            if (forced_tool_name.empty()) {
                throw std::runtime_error("tool_choice function name must not be empty");
            }
            chat_tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            parallel_tool_calls = false;
        } else {
            throw std::runtime_error("invalid tool_choice");
        }
    }

    if (chat_tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED && !tools_enabled) {
        throw std::runtime_error("tool_choice requires a non-empty tools array");
    }

    if (!forced_tool_name.empty()) {
        json filtered_tools = json::array();
        for (const auto & tool : tools_for_prompt) {
            if (tool.contains("function") && json_value(tool["function"], "name", std::string()) == forced_tool_name) {
                filtered_tools.push_back(tool);
            }
        }
        if (filtered_tools.empty()) {
            throw std::runtime_error("Unknown forced tool name: " + forced_tool_name);
        }
        tools_for_prompt = filtered_tools;
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    // Handle "response_format" field
    if (body.contains("response_format")) {
        json response_format      = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            json_schema = json_value(response_format, "schema", json::object());
        } else if (response_type == "json_schema") {
            json json_schema_wrapper = json_value(response_format, "json_schema", json::object());
            json_schema = json_value(json_schema_wrapper, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::runtime_error("response_format type must be one of \"text\" or \"json_object\", but got: " + response_type);
        }
    }

    if (tools_enabled && chat_tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && !grammar.empty()) {
        throw std::runtime_error("grammar cannot be used together with tool calls");
    }

    common_chat_params chat_params = format_chat(
        model,
        chat_template,
        body.at("messages"),
        tools_for_prompt,
        chat_tool_choice,
        parallel_tool_calls,
        json_schema.is_null() || json_schema.empty() ? "" : json_schema.dump(),
        grammar);

    const bool tool_calling = tools_enabled && chat_tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    llama_params["prompt"] = chat_params.prompt;
    // The BitNet vendored core predates upstream lazy/tool grammar support; keep
    // the upstream-rendered prompt and parser, but do not feed tool grammars to
    // the old sampler.
    if (!tool_calling && !chat_params.grammar.empty() && !chat_params.grammar_lazy) {
        llama_params["grammar"] = chat_params.grammar;
    }
    if (tools_enabled) {
        llama_params["tools"] = tools_for_prompt;
    }
    if (!chat_params.parser.empty()) {
        llama_params["chat_parser"] = chat_params.parser;
        llama_params["chat_format"] = static_cast<int>(chat_params.format);
        llama_params["generation_prompt"] = chat_params.generation_prompt;
    }
    for (const auto & stop : chat_params.additional_stops) {
        llama_params["stop"].push_back(stop);
    }

    // Handle "n" field
    int n_choices = json_value(body, "n", 1);
    if (n_choices != 1) {
        throw std::runtime_error("Only one completion choice is allowed");
    }

    // Handle "logprobs" field
    // TODO: The response format of this option is not yet OAI-compatible, but seems like no one really using it; We may need to fix it in the future
    if (json_value(body, "logprobs", false)) {
        llama_params["n_probs"] = json_value(body, "top_logprobs", 20);
    } else if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        throw std::runtime_error("top_logprobs requires logprobs to be set to true");
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", "tfs_z",... via OAI endpoint.
    // See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

static json format_final_response_oaicompat(const json & request, const json & result, const std::string & completion_id, bool streaming = false, bool verbose = false) {
    bool stopped_word        = json_value(result, "stopped_word", false);
    bool stopped_eos         = json_value(result, "stopped_eos", false);
    int num_tokens_predicted = json_value(result, "tokens_predicted", 0);
    int num_prompt_tokens    = json_value(result, "tokens_evaluated", 0);
    std::string content      = json_value(result, "content", std::string(""));
    parsed_tool_call_content parsed_tool_calls = parse_tool_calls_for_request(content, request);
    if (parsed_tool_calls.has_tool_calls) {
        content = parsed_tool_calls.content;
    }

    std::string finish_reason = "length";
    if (parsed_tool_calls.has_tool_calls) {
        finish_reason = "tool_calls";
    } else if (stopped_word || stopped_eos) {
        finish_reason = "stop";
    }

    json message = {
        {"content", content},
        {"role", "assistant"},
    };
    if (parsed_tool_calls.has_tool_calls) {
        message["content"] = content.empty() ? json(nullptr) : json(content);
        message["tool_calls"] = parsed_tool_calls.tool_calls;
    }

    json delta = json::object();
    if (parsed_tool_calls.has_tool_calls) {
        delta["role"] = "assistant";
        delta["tool_calls"] = parsed_tool_calls.tool_calls;
    }

    json choices =
        streaming ? json::array({json{{"finish_reason", finish_reason},
                                        {"index", 0},
                                        {"delta", delta}}})
                  : json::array({json{{"finish_reason", finish_reason},
                                        {"index", 0},
                                        {"message", message}}});

    std::time_t t = std::time(0);

    json res = json {
        {"choices", choices},
        {"created", t},
        {"model",
            json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", streaming ? "chat.completion.chunk" : "chat.completion"},
        {"usage", json {
            {"completion_tokens", num_tokens_predicted},
            {"prompt_tokens",     num_prompt_tokens},
            {"total_tokens",      num_tokens_predicted + num_prompt_tokens}
        }},
        {"id", completion_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = result;
    }

    if (result.contains("completion_probabilities")) {
        res["completion_probabilities"] = json_value(result, "completion_probabilities", json::array());
    }

    return res;
}

// return value is vector as there is one case where we might need to generate two responses
static std::vector<json> format_partial_response_oaicompat(const json & result, const std::string & completion_id, const json & request = json::object()) {
    if (!result.contains("model") || !result.contains("oaicompat_token_ctr")) {
        return std::vector<json>({result});
    }

    bool first = json_value(result, "oaicompat_token_ctr", 0) == 0;
    std::string modelname = json_value(result, "model", std::string(DEFAULT_OAICOMPAT_MODEL));

    bool stopped_word   = json_value(result, "stopped_word",  false);
    bool stopped_eos    = json_value(result, "stopped_eos",   false);
    bool stopped_limit  = json_value(result, "stopped_limit", false);
    std::string content = json_value(result, "content",       std::string(""));
    parsed_tool_call_content parsed_tool_calls = parse_tool_calls_for_request(content, request);

    std::string finish_reason;
    if (parsed_tool_calls.has_tool_calls) {
        finish_reason = "tool_calls";
    } else if (stopped_word || stopped_eos) {
        finish_reason = "stop";
    }
    if (stopped_limit) {
        finish_reason = "length";
    }

    std::time_t t = std::time(0);

    json choices;

    if (parsed_tool_calls.has_tool_calls) {
        choices = json::array({json{{"finish_reason", finish_reason},
                                    {"index", 0},
                                    {"delta", json{
                                        {"role", "assistant"},
                                        {"tool_calls", parsed_tool_calls.tool_calls},
                                    }}}});
    } else if (!finish_reason.empty() && !content.empty()) {
        choices = json::array({json{{"finish_reason", finish_reason},
                                    {"index", 0},
                                    {"delta", json{{"content", content}}}}});
    } else if (!finish_reason.empty()) {
        choices = json::array({json{{"finish_reason", finish_reason},
                                    {"index", 0},
                                    {"delta", json::object()}}});
    } else {
        if (first) {
            if (content.empty()) {
                choices = json::array({json{{"finish_reason", nullptr},
                                            {"index", 0},
                                            {"delta", json{{"role", "assistant"}}}}});
            } else {
                // We have to send this as two updates to conform to openai behavior
                json initial_ret = json{{"choices", json::array({json{
                                        {"finish_reason", nullptr},
                                        {"index", 0},
                                        {"delta", json{
                                            {"role", "assistant"}
                                        }}}})},
                            {"created", t},
                            {"id", completion_id},
                            {"model", modelname},
                            {"object", "chat.completion.chunk"}};

                json second_ret = json{
                            {"choices", json::array({json{{"finish_reason", nullptr},
                                                            {"index", 0},
                                                            {"delta", json{
                                                            {"content", content}}}
                                                            }})},
                            {"created", t},
                            {"id", completion_id},
                            {"model", modelname},
                            {"object", "chat.completion.chunk"}};

                return std::vector<json>({initial_ret, second_ret});
            }
        } else {
            // Some idiosyncrasy in task processing logic makes several trailing calls
            // with empty content, we ignore these at the calee site.
            if (content.empty()) {
                return std::vector<json>({json::object()});
            }

            choices = json::array({json{
                {"finish_reason", nullptr},
                {"index", 0},
                {"delta",
                json{
                    {"content", content},
                }},
            }});
        }
    }

    json ret = json {
        {"choices", choices},
        {"created", t},
        {"id",      completion_id},
        {"model",   modelname},
        {"object",  "chat.completion.chunk"}
    };
    if (!finish_reason.empty()) {
        int num_tokens_predicted = json_value(result, "tokens_predicted", 0);
        int num_prompt_tokens    = json_value(result, "tokens_evaluated", 0);
        ret.push_back({"usage", json {
            {"completion_tokens", num_tokens_predicted},
            {"prompt_tokens",     num_prompt_tokens},
            {"total_tokens",      num_tokens_predicted + num_prompt_tokens}
        }});
    }

    return std::vector<json>({ret});
}

static json format_embeddings_response_oaicompat(const json & request, const json & embeddings) {
    json data = json::array();
    int i = 0;
    for (const auto & elem : embeddings) {
        data.push_back(json{
            {"embedding", json_value(elem, "embedding", json::array())},
            {"index",     i++},
            {"object",    "embedding"}
        });
    }

    json res = json {
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"usage", json { // TODO: fill
            {"prompt_tokens", 0},
            {"total_tokens", 0}
        }},
        {"data", data}
    };

    return res;
}

static json format_response_rerank(const json & request, const json & ranks) {
    json data = json::array();
    int i = 0;
    for (const auto & rank : ranks) {
        data.push_back(json{
            {"index",    i++},
            {"relevance_score", json_value(rank, "score", 0.0)},
        });
    }

    json res = json {
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"usage", json { // TODO: fill
            {"prompt_tokens", 0},
            {"total_tokens", 0}
        }},
        {"results", data}
    };

    return res;
}

static bool is_valid_utf8(const std::string & str) {
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str.data());
    const unsigned char* end = bytes + str.length();

    while (bytes < end) {
        if (*bytes <= 0x7F) {
            // 1-byte sequence (0xxxxxxx)
            bytes++;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // 2-byte sequence (110xxxxx 10xxxxxx)
            if (end - bytes < 2 || (bytes[1] & 0xC0) != 0x80)
                return false;
            bytes += 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 3 || (bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80)
                return false;
            bytes += 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            if (end - bytes < 4 || (bytes[1] & 0xC0) != 0x80 ||
                (bytes[2] & 0xC0) != 0x80 || (bytes[3] & 0xC0) != 0x80)
                return false;
            bytes += 4;
        } else {
            // Invalid UTF-8 lead byte
            return false;
        }
    }

    return true;
}

static json format_tokenizer_response(const json & tokens) {
    return json {
        {"tokens", tokens}
    };
}

static json format_detokenized_response(const std::string & content) {
    return json {
        {"content", content}
    };
}

static json format_error_response(const std::string & message, const enum error_type type) {
    std::string type_str;
    int code = 500;
    switch (type) {
        case ERROR_TYPE_INVALID_REQUEST:
            type_str = "invalid_request_error";
            code = 400;
            break;
        case ERROR_TYPE_AUTHENTICATION:
            type_str = "authentication_error";
            code = 401;
            break;
        case ERROR_TYPE_NOT_FOUND:
            type_str = "not_found_error";
            code = 404;
            break;
        case ERROR_TYPE_SERVER:
            type_str = "server_error";
            code = 500;
            break;
        case ERROR_TYPE_PERMISSION:
            type_str = "permission_error";
            code = 403;
            break;
        case ERROR_TYPE_NOT_SUPPORTED:
            type_str = "not_supported_error";
            code = 501;
            break;
        case ERROR_TYPE_UNAVAILABLE:
            type_str = "unavailable_error";
            code = 503;
            break;
    }
    return json {
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}
