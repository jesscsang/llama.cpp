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

static std::string model_token_to_piece(const struct llama_model * model, llama_token token, bool special = true) {
    std::string piece;
    piece.resize(16);

    int32_t n_chars = llama_token_to_piece(model, token, &piece[0], (int32_t) piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int32_t check = llama_token_to_piece(model, token, &piece[0], (int32_t) piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    } else {
        piece.resize(n_chars);
    }

    return piece;
}

static std::string message_content_to_string(const json & curr_msg) {
    if (!curr_msg.contains("content") || curr_msg["content"].is_null()) {
        return "";
    }

    const json & content_json = curr_msg["content"];
    if (content_json.is_string()) {
        return content_json.get<std::string>();
    }

    if (content_json.is_array()) {
        std::string content;
        for (const auto & part : content_json) {
            if (part.contains("text") && part["text"].is_string()) {
                if (!content.empty()) {
                    content += "\n";
                }
                content += part["text"].get<std::string>();
            }
        }
        return content;
    }

    throw std::runtime_error("Invalid 'content' type (ref: https://github.com/ggerganov/llama.cpp/issues/8367)");
}

// Format given chat without tool metadata. If tmpl is empty, we take the template from model metadata.
inline std::string format_chat_legacy(const struct llama_model * model, const std::string & tmpl, const std::vector<json> & messages) {
    std::vector<common_chat_msg> chat;

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & curr_msg = messages[i];

        std::string role = json_value(curr_msg, "role", std::string(""));
        if (!curr_msg.contains("content")) {
            throw std::runtime_error("Missing 'content' (ref: https://github.com/ggerganov/llama.cpp/issues/8367)");
        }

        std::string content = message_content_to_string(curr_msg);
        chat.push_back({role, content});
    }

    const auto formatted_chat = common_chat_apply_template(model, tmpl, chat, true);
    LOG_DBG("formatted_chat: '%s'\n", formatted_chat.c_str());

    return formatted_chat;
}

static std::string llama_get_chat_template(const struct llama_model * model) {
    std::string template_key = "tokenizer.chat_template";
    // call with NULL buffer to get the total size of the string
    int32_t res = llama_model_meta_val_str(model, template_key.c_str(), NULL, 0);
    if (res < 0) {
        return "";
    } else {
        std::vector<char> model_template(res, 0);
        llama_model_meta_val_str(model, template_key.c_str(), model_template.data(), model_template.size());
        return std::string(model_template.data(), model_template.size());
    }
}

static bool json_contains_tool_calls(const json & messages) {
    if (!messages.is_array()) {
        return false;
    }

    for (const auto & message : messages) {
        if (message.contains("tool_calls") || json_value(message, "role", std::string()) == "tool") {
            return true;
        }
    }

    return false;
}

static std::string tool_call_system_prompt(const json & tools) {
    std::ostringstream ss;
    ss << "You can call functions. When using a function, respond only with "
       << "<tool_call>[{\"name\":\"function_name\",\"arguments\":{}}]</tool_call>.\n"
       << "Available tools: " << tools.dump(-1, ' ', false, json::error_handler_t::replace) << "\n";
    return ss.str();
}

static std::string format_chat_falcon_tools(
        const struct llama_model * model,
        const std::vector<json> & messages,
        const json & tools) {
    std::ostringstream ss;

    const std::string eos_token = model_token_to_piece(model, llama_token_eos(model), true);
    const bool has_tools = tools.is_array() && !tools.empty();

    size_t first_message = 0;
    const bool has_system_message = !messages.empty() && json_value(messages[0], "role", std::string()) == "system";
    if (has_tools || has_system_message) {
        ss << "<|system|>\n";
    }
    if (has_system_message) {
        const std::string system_content = message_content_to_string(messages[0]);
        ss << system_content;
        if (!system_content.empty() && system_content.back() != '\n') {
            ss << "\n";
        }
        first_message = 1;
    }
    if (has_tools) {
        ss << tool_call_system_prompt(tools);
    }

    for (size_t i = first_message; i < messages.size(); ++i) {
        const auto & message = messages[i];
        const std::string role = json_value(message, "role", std::string());
        const std::string content = message_content_to_string(message);

        if (role == "user") {
            ss << "<|user|>\n" << content << "\n";
        } else if (role == "assistant") {
            const bool has_tool_calls = message.contains("tool_calls") && message["tool_calls"].is_array() && !message["tool_calls"].empty();
            if (!content.empty() || has_tool_calls) {
                ss << "<|assistant|>\n";
            }
            if (!content.empty()) {
                ss << content;
            }
            if (has_tool_calls) {
                ss << "\n<tool_call>\n" << message["tool_calls"].dump(2) << "\n</tool_call>";
            }
            ss << eos_token << "\n";
        } else if (role == "tool") {
            ss << "<|assistant|>\n<tool_response>\n" << content << "\n</tool_response>\n";
        } else if (role == "system") {
            ss << "<|system|>\n" << content << "\n";
        } else if (!content.empty()) {
            ss << "<|" << role << "|>\n" << content << "\n";
        }
    }

    ss << "<|assistant|>\n";

    const auto formatted_chat = ss.str();
    LOG_DBG("formatted_chat tools: '%s'\n", formatted_chat.c_str());
    return formatted_chat;
}

static std::string format_chat_with_tool_prompt(
        const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<json> & messages,
        const json & tools) {
    std::vector<json> patched_messages = messages;
    const std::string prompt = tool_call_system_prompt(tools);

    if (!patched_messages.empty() && json_value(patched_messages[0], "role", std::string()) == "system") {
        patched_messages[0]["content"] = message_content_to_string(patched_messages[0]) + "\n\n" + prompt;
    } else {
        json system_msg = {
            {"role", "system"},
            {"content", prompt},
        };
        patched_messages.insert(patched_messages.begin(), system_msg);
    }

    return format_chat_legacy(model, tmpl, patched_messages);
}

// Format given chat. If tools are present, render the tool-aware Falcon template
// used by the BitNet/Falcon3 model metadata; otherwise fall back to the legacy
// chat-template wrapper.
inline std::string format_chat(
        const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<json> & messages,
        const json & tools = json()) {
    const bool has_tools = tools.is_array() && !tools.empty();
    const bool has_tool_history = json_contains_tool_calls(messages);
    if (!has_tools && !has_tool_history) {
        return format_chat_legacy(model, tmpl, messages);
    }

    const std::string tmpl_src = tmpl.empty() ? llama_get_chat_template(model) : tmpl;
    const bool is_falcon_tool_template =
        tmpl_src.find("<|assistant|>") != std::string::npos &&
        tmpl_src.find("<tool_call>")   != std::string::npos &&
        tmpl_src.find("<tools>")       != std::string::npos;

    if ((has_tools || has_tool_history) && is_falcon_tool_template) {
        return format_chat_falcon_tools(model, messages, tools);
    }

    if (has_tools) {
        return format_chat_with_tool_prompt(model, tmpl, messages, tools);
    }

    return format_chat_legacy(model, tmpl, messages);
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

static std::string gbnf_literal(const std::string & text) {
    std::string out = "\"";
    for (char c : text) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    out += "\"";
    return out;
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

static std::string build_tool_call_grammar(const json & tools, const std::string & forced_name, bool parallel_tool_calls) {
    const auto names = tool_names_from_request(tools, forced_name);
    if (names.empty()) {
        throw std::runtime_error(forced_name.empty() ? "No function tools were provided" : "Unknown forced tool name: " + forced_name);
    }

    std::ostringstream name_rule;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0) {
            name_rule << " | ";
        }
        name_rule << gbnf_literal(json(names[i]).dump());
    }

    const std::string repeat_rule = parallel_tool_calls ? " (\",\" space tool-call)*" : "";

    std::ostringstream grammar;
    grammar
        << "root ::= \"<tool_call>\" space \"[\" space tool-call" << repeat_rule << " \"]\" space \"</tool_call>\" space\n"
        << "tool-call ::= \"{\" space \"\\\"name\\\"\" space \":\" space tool-name space \",\" space "
        << "\"\\\"arguments\\\"\" space \":\" space object \"}\" space\n"
        << "tool-name ::= " << name_rule.str() << "\n"
        << "value ::= object | array | string | number | boolean | null\n"
        << "object ::= \"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? \"}\" space\n"
        << "array ::= \"[\" space ( value (\",\" space value)* )? \"]\" space\n"
        << "string ::= \"\\\"\" char* \"\\\"\" space\n"
        << "char ::= [^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})\n"
        << "number ::= (\"-\"? integral-part) (\".\" [0-9]+)? ([eE] [-+]? integral-part)? space\n"
        << "integral-part ::= \"0\" | [1-9] [0-9]*\n"
        << "boolean ::= (\"true\" | \"false\") space\n"
        << "null ::= \"null\" space\n"
        << "space ::= [ \\t\\n\\r]*\n";

    return grammar.str();
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
    bool force_tool_call = false;
    bool parallel_tool_calls = json_value(body, "parallel_tool_calls", true);
    std::string forced_tool_name;

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
            } else if (choice == "required") {
                force_tool_call = true;
            } else if (choice != "auto") {
                throw std::runtime_error("tool_choice must be one of \"none\", \"auto\", \"required\", or a function choice object");
            }
        } else if (tool_choice.is_object()) {
            if (json_value(tool_choice, "type", std::string()) != "function" || !tool_choice.contains("function")) {
                throw std::runtime_error("tool_choice object must be of type \"function\"");
            }
            forced_tool_name = json_value(tool_choice["function"], "name", std::string());
            if (forced_tool_name.empty()) {
                throw std::runtime_error("tool_choice function name must not be empty");
            }
            force_tool_call = true;
            parallel_tool_calls = false;
        } else {
            throw std::runtime_error("invalid tool_choice");
        }
    }

    if (force_tool_call && !tools_enabled) {
        throw std::runtime_error("tool_choice requires a non-empty tools array");
    }

    // Apply chat template to the list of messages
    llama_params["prompt"] = format_chat(model, chat_template, body.at("messages"), tools_for_prompt);

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
            llama_params["json_schema"] = json_value(response_format, "schema", json::object());
        } else if (response_type == "json_schema") {
            json json_schema = json_value(response_format, "json_schema", json::object());
            llama_params["json_schema"] = json_value(json_schema, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::runtime_error("response_format type must be one of \"text\" or \"json_object\", but got: " + response_type);
        }
    }

    if (force_tool_call) {
        if (llama_params.contains("json_schema")) {
            throw std::runtime_error("response_format cannot be used together with required tool calls");
        }
        if (body.contains("grammar") && !body.at("grammar").is_null()) {
            throw std::runtime_error("grammar cannot be used together with required tool calls");
        }
        llama_params["grammar"] = build_tool_call_grammar(tools_for_prompt, forced_tool_name, parallel_tool_calls);
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
    const bool request_has_tools =
        request.contains("tools") && request["tools"].is_array() && !request["tools"].empty() &&
        json_value(request, "tool_choice", std::string("auto")) != "none";
    parsed_tool_call_content parsed_tool_calls = parse_tool_calls_from_content(
        content,
        request_has_tools ? tool_names_from_request(request["tools"]) : std::vector<std::string>(),
        request_has_tools);
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
    const bool request_has_tools =
        request.contains("tools") && request["tools"].is_array() && !request["tools"].empty() &&
        json_value(request, "tool_choice", std::string("auto")) != "none";
    parsed_tool_call_content parsed_tool_calls = parse_tool_calls_from_content(
        content,
        request_has_tools ? tool_names_from_request(request["tools"]) : std::vector<std::string>(),
        request_has_tools);

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
