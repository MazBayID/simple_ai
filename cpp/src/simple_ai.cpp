#include "simple_ai/simple_ai.hpp"

#include <random>
#include <stdexcept>
#include <vector>

namespace simple_ai {

void TinyTransitionModel::train(const std::string& text) {
    if (text.empty()) {
        throw std::invalid_argument("training text must not be empty");
    }

    counts_.clear();
    initial_counts_.clear();

    char previous = '\0';
    for (const char current : text) {
        if (previous == '\0') {
            ++initial_counts_[current];
        } else {
            ++counts_[previous][current];
        }
        previous = current;
    }
    trained_ = true;
}

std::string TinyTransitionModel::generate(
    const std::string& prompt,
    const std::size_t max_new_tokens,
    const unsigned int seed
) const {
    if (!trained_) {
        throw std::logic_error("model is not trained");
    }

    std::string output = prompt;
    std::mt19937 generator(seed);
    char previous = output.empty() ? '\0' : output.back();

    for (std::size_t i = 0; i < max_new_tokens; ++i) {
        const auto found = counts_.find(previous);

        if (found == counts_.end() || found->second.empty()) {
            if (initial_counts_.empty()) {
                break;
            }
            std::vector<char> chars;
            std::vector<std::size_t> weights;
            for (const auto& entry : initial_counts_) {
                chars.push_back(entry.first);
                weights.push_back(entry.second);
            }
            std::discrete_distribution<std::size_t> distribution(
                weights.begin(), weights.end()
            );
            previous = chars[distribution(generator)];
        } else {
            std::vector<char> chars;
            std::vector<std::size_t> weights;
            for (const auto& entry : found->second) {
                chars.push_back(entry.first);
                weights.push_back(entry.second);
            }
            std::discrete_distribution<std::size_t> distribution(
                weights.begin(), weights.end()
            );
            previous = chars[distribution(generator)];
        }

        output.push_back(previous);
    }

    return output;
}

bool TinyTransitionModel::trained() const noexcept {
    return trained_;
}

}  // namespace simple_ai
