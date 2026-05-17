// Chat template renderer implementation.

#include "chat_template.h"

namespace dflash27b {

ChatFormat chat_format_for_arch(const std::string & arch) {
    if (arch == "laguna") return ChatFormat::LAGUNA;
    // qwen35, qwen3, gemma4 all use the Qwen3/ChatML format
    return ChatFormat::QWEN3;
}

std::string render_chat_template(
    const std::vector<ChatMessage> & messages,
    ChatFormat format,
    bool add_generation_prompt,
    bool enable_thinking)
{
    std::string result;

    switch (format) {
    case ChatFormat::QWEN3: {
        // Qwen3/3.5 ChatML format:
        //   <|im_start|>system\nYou are...<|im_end|>\n
        //   <|im_start|>user\nHello<|im_end|>\n
        //   <|im_start|>assistant\n...
        for (const auto & msg : messages) {
            result += "<|im_start|>";
            result += msg.role;
            result += '\n';
            result += msg.content;
            result += "<|im_end|>\n";
        }
        if (add_generation_prompt) {
            result += "<|im_start|>assistant\n";
            if (!enable_thinking) {
                // Qwen3 thinking disabled: inject closed think block so the
                // model skips reasoning and generates the answer directly.
                result += "<think>\n\n</think>\n\n";
            }
        }
        break;
    }

    case ChatFormat::LAGUNA: {
        // Laguna/DeepSeek format:
        //   <｜begin▁of▁sentence｜>system content
        //   <｜User｜>user content<｜Assistant｜>
        result = "<｜begin▁of▁sentence｜>";
        for (const auto & msg : messages) {
            if (msg.role == "system") {
                result += msg.content;
            } else if (msg.role == "user") {
                result += "<｜User｜>";
                result += msg.content;
            } else if (msg.role == "assistant") {
                result += "<｜Assistant｜>";
                result += msg.content;
            } else if (msg.role == "tool") {
                // Tool results inline as user content
                result += "<｜User｜>[tool_result:" + msg.tool_call_id + "]\n";
                result += msg.content;
            }
        }
        if (add_generation_prompt) {
            result += "<｜Assistant｜>";
        }
        break;
    }
    }

    return result;
}

}  // namespace dflash27b
