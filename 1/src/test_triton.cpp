#include <http_client.h>
#include <grpc_client.h>

#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace triton::client;

void PrintUsage(const char *program_name) {
    std::cerr << "Usage: " << program_name << " [-i <protocol>] [-m <model_name>] [-u <url>]" << std::endl;
}

bool ParseArgs(int argc, char *argv[], std::string &protocol, std::string &model_name, std::string &url) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" || arg == "-m" || arg == "-u") {
            if ((i + 1) >= argc) {
                std::cerr << "Missing value for argument: " << arg << std::endl;
                return false;
            }

            std::string value = argv[++i];
            if (arg == "-i") {
                protocol = value;
            } else if (arg == "-m") {
                model_name = value;
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

    if (protocol != "grpc" && protocol != "http") {
        std::cerr << "Unknown protocol: " << protocol << " (expected 'http' or 'grpc')" << std::endl;
        return false;
    }

    if (url.empty()) {
        url = protocol == "grpc" ? "localhost:8001" : "localhost:8000";
    }

    return true;
}

template <typename ClientType>
class TritonClient {
public:
    TritonClient(const std::string &url) {
        Error err = ClientType::Create(&client_, url);
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
    ~TritonClient() noexcept = default;

    void Infer(const std::string &model_name) {
        std::cout << "Running inference with model: " << model_name << std::endl;

        CreateInputs(model_name);
        CreateInferRequestedOutput("output");

        InferResult *result;
        InferOptions options(model_name);
        Error err = client_->Infer(&result, options, inputs_, outputs_);
        if (!err.IsOk()) {
            throw std::runtime_error("Inference failed: " + err.Message());
        }
        std::unique_ptr<InferResult> result_ptr(result);
        std::cout << "Inference successful!" << std::endl;
    }

private:
    static size_t GetDataTypeSize(const std::string &datatype) {
        if (datatype == "BOOL") return 1;
        if (datatype == "INT8" || datatype == "UINT8") return 1;
        if (datatype == "INT16" || datatype == "UINT16") return 2;
        if (datatype == "INT32" || datatype == "UINT32" || datatype == "FP32") return 4;
        if (datatype == "INT64" || datatype == "UINT64" || datatype == "FP64") return 8;
        throw std::invalid_argument("Unknown datatype: " + datatype);
    }

    void CreateInferInput(const std::string &name, const std::vector<int64_t> &dims, const std::string &datatype) {
        InferInput *input;
        Error err = InferInput::Create(&input, name, dims, datatype);
        if (!err.IsOk()) {
            throw std::runtime_error("Failed to create InferInput: " + err.Message());
        }
        inputs_.emplace_back(input);
        input_ptrs_.emplace_back(std::unique_ptr<InferInput>(input));

        size_t input_size = GetDataTypeSize(datatype) * std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
        auto data = std::make_unique<uint8_t[]>(input_size);
        std::fill(data.get(), data.get() + input_size, 0);
        err = input->AppendRaw(data.get(), input_size);
        if (!err.IsOk()) {
            throw std::runtime_error("Failed to append data to InferInput: " + err.Message());
        }
        input_data_.emplace_back(std::move(data));
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

    void CreateInputs(const std::string &model_name) {
        const int seq_len = 10, num_layers = 4, input_size = 128, hidden_size = 256;
        if (model_name == "resnet50" || model_name == "mobilenet_v2") {
            CreateInferInput("input", {1, 3, 224, 224}, "FP32");
        } else if (model_name == "yolov5s") {
            CreateInferInput("input", {1, 3, 640, 640}, "FP32");
        } else if (model_name == "bert-base-cased") {
            CreateInferInput("input_ids", {1, 32}, "INT64");
            CreateInferInput("attention_mask", {1, 32}, "INT64");
            CreateInferInput("token_type_ids", {1, 32}, "INT64");
        } else if (model_name == "rnn" || model_name == "gru") {
            CreateInferInput("input", {seq_len, 1, input_size}, "FP32");
            CreateInferInput("h0", {num_layers, 1, hidden_size}, "FP32");
        } else if (model_name == "lstm") {
            CreateInferInput("input", {seq_len, 1, input_size}, "FP32");
            CreateInferInput("h0", {num_layers, 1, hidden_size}, "FP32");
            CreateInferInput("c0", {num_layers, 1, hidden_size}, "FP32");
        } else {
            throw std::invalid_argument("Unknown model: " + model_name);
        }
    }

    std::unique_ptr<ClientType> client_;
    std::vector<InferInput*> inputs_;
    std::vector<const InferRequestedOutput*> outputs_;
    std::vector<std::unique_ptr<InferInput>> input_ptrs_;
    std::vector<std::unique_ptr<InferRequestedOutput>> output_ptrs_;
    std::vector<std::unique_ptr<uint8_t[]>> input_data_;
};

int main(int argc, char *argv[]) {
    std::string protocol = "http", model_name, url;

    if (!ParseArgs(argc, argv, protocol, model_name, url)) {
        PrintUsage(argv[0]);
        return 1;
    }

    if (protocol == "grpc") {
        TritonClient<InferenceServerGrpcClient> client(url);
        client.Infer(model_name);
    } else {
        TritonClient<InferenceServerHttpClient> client(url);
        client.Infer(model_name);
    }
    return 0;
}