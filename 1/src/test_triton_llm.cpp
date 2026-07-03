#include <http_client.h>
#include <grpc_client.h>

#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace triton::client;

void PrintUsage(const char *program_name) {
    std::cerr << "Usage: " << program_name << " [-m <model_name>] [-u <url>] [-p <prompt>]" << std::endl;
}

bool ParseArgs(int argc, char *argv[], std::string &model_name, std::string &url, std::string &prompt) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" || arg == "-m" || arg == "-u" || arg == "-p") {
            if ((i + 1) >= argc) {
                std::cerr << "Missing value for argument: " << arg << std::endl;
                return false;
            }

            std::string value = argv[++i];
            if (arg == "-m") {
                model_name = value;
            } else if (arg == "-p") {
                prompt = value;
            } else {
                url = value;
            }
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }

    if (model_name.empty()) {
        return false;
    }

    return true;
}

class TritonLLMClient {
public:
    TritonLLMClient(const std::string &url) {
        Error err = InferenceServerGrpcClient::Create(&client_, url);
        if (!err.IsOk()) {
            throw std::runtime_error("Failed to create Triton client: " + err.Message());
        }
        bool server_ready;
        err = client_->IsServerReady(&server_ready);
        if (!err.IsOk()) {
            throw std::runtime_error("Failed to check server readiness: " + err.Message());
        }
        if (!server_ready) {
            throw std::runtime_error("Server is not ready!");
        }
    }
    ~TritonLLMClient() noexcept = default;

    void Infer(const std::string &model_name, const std::string &prompt) {
        std::cout << "Running inference with model: " << model_name << std::endl;
        std::cout << "Prompt: " << prompt << std::endl;

        std::atomic<bool> finished{false};
        auto callback = [&finished](InferResult* result) {
            std::unique_ptr<InferResult> result_ptr(result);
            std::vector<std::string> text_output;
            Error err = result->StringData("text_output", &text_output);
            if (!err.IsOk()) {
                std::cerr << "Failed to get text_output: " << err.Message() << std::endl;
                return;
            }
            for (const auto& output : text_output) {
                std::cout << output;
            }
            std::cout.flush();

            std::vector<std::string> finish_reason;
            err = result->StringData("finish_reason", &finish_reason);
            if (!err.IsOk()) {
                std::cerr << "Failed to get finish_reason: " << err.Message() << std::endl;
                return;
            }
            if (!finish_reason.empty() && finish_reason[0] != "None") {
                std::cout << std::endl;
                finished.store(true);
                finished.notify_all();
            }
        };
        client_->StartStream(callback);

        CreateInputs(model_name, prompt);
        CreateInferRequestedOutput("text_output");
        CreateInferRequestedOutput("finish_reason");
        InferOptions options(model_name);
        Error err = client_->AsyncStreamInfer(options, inputs_, outputs_);
        if (!err.IsOk()) {
            throw std::runtime_error("Inference failed: " + err.Message());
        }
        std::cout << "Response: ";
        if (model_name.starts_with("DeepSeek-R1")) {
            std::cout << "<think>\n";
        }
        std::cout.flush();

        finished.wait(false);
        std::cout << "Inference successful!" << std::endl;

        client_->StopStream();
    }

private:
    void CreateInferInput(const std::string &name, const std::vector<int64_t> &dims, const std::string &datatype) {
        InferInput *input;
        Error err = InferInput::Create(&input, name, dims, datatype);
        if (!err.IsOk()) {
            throw std::runtime_error("Failed to create InferInput: " + err.Message());
        }
        inputs_.emplace_back(input);
        input_ptrs_.emplace_back(std::unique_ptr<InferInput>(input));
    }

    void CreateInferRequestedOutput(const std::string &name) {
        InferRequestedOutput *output;
        Error err = InferRequestedOutput::Create(&output, name);
        if (!err.IsOk()) {
            throw std::runtime_error("Failed to create InferRequestedOutput: " + err.Message());
        }
        outputs_.emplace_back(output);
        output_ptrs_.emplace_back(std::unique_ptr<InferRequestedOutput>(output));
    }

    void CreateInputs(const std::string &model_name, const std::string &prompt) {
        const int max_tokens = 512, seed = 42;
        CreateInferInput("text_input", {1}, "BYTES");
        std::string text_input;
        if (model_name.starts_with("Qwen3")) {
            text_input = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        } else if (model_name.starts_with("Qwen2.5")) {
            text_input = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        } else if (model_name.starts_with("DeepSeek-R1")) {
            text_input = "<｜begin▁of▁sentence｜><｜User｜>" + prompt + "<｜Assistant｜><think>\n";
        } else {
            throw std::invalid_argument("Unknown model: " + model_name);
        }
        inputs_.back()->AppendFromString({text_input});

        CreateInferInput("sampling_parameters", {1}, "BYTES");
        std::string sampling_parameters = "{\"max_tokens\":" + std::to_string(max_tokens) + ",\"seed\":" + std::to_string(seed) + "}";
        inputs_.back()->AppendFromString({sampling_parameters});

        CreateInferInput("stream", {1}, "BOOL");
        auto stream = std::make_unique<bool>(true);
        inputs_.back()->AppendRaw(reinterpret_cast<uint8_t*>(stream.get()), sizeof(bool));
        input_data_.emplace_back(std::move(stream));

        CreateInferInput("exclude_input_in_output", {1}, "BOOL");
        auto exclude_input_in_output = std::make_unique<bool>(true);
        inputs_.back()->AppendRaw(reinterpret_cast<uint8_t*>(exclude_input_in_output.get()), sizeof(bool));
        input_data_.emplace_back(std::move(exclude_input_in_output));

        CreateInferInput("return_finish_reason", {1}, "BOOL");
        auto return_finish_reason = std::make_unique<bool>(true);
        inputs_.back()->AppendRaw(reinterpret_cast<uint8_t*>(return_finish_reason.get()), sizeof(bool));
        input_data_.emplace_back(std::move(return_finish_reason));
    }

    std::unique_ptr<InferenceServerGrpcClient> client_;
    std::vector<InferInput*> inputs_;
    std::vector<const InferRequestedOutput*> outputs_;
    std::vector<std::unique_ptr<InferInput>> input_ptrs_;
    std::vector<std::unique_ptr<InferRequestedOutput>> output_ptrs_;
    std::vector<std::unique_ptr<bool>> input_data_;
};

int main(int argc, char *argv[]) {
    std::string model_name, url = "localhost:8001", prompt = "Hello!";

    if (!ParseArgs(argc, argv, model_name, url, prompt)) {
        PrintUsage(argv[0]);
        return 1;
    }

    TritonLLMClient client(url);
    client.Infer(model_name, prompt);

    return 0;
}