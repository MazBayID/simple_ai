#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>

namespace simple_ai {

class TinyTransitionModel {
public:
    void train(const std::string& text);
    std::string generate(
        const std::string& prompt,
        std::size_t max_new_tokens,
        unsigned int seed = 7
    ) const;
    bool trained() const noexcept;

private:
    std::unordered_map<char, std::unordered_map<char, std::size_t>> counts_;
    std::unordered_map<char, std::size_t> initial_counts_;
    bool trained_ = false;
};

}  // namespace simple_ai
