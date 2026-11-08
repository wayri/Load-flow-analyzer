#pragma once
#include "io.hpp"
#include "default_method.hpp"
#include <iostream>

namespace lf {
inline void usage() {
    std::cout << "Load flow analyzer | default method: " << default_method << "\n\n"
        "load_flow --buses buses.csv --branches branches.csv [options]\n"
        "  --method gs|nr       Select Gauss-Seidel or Newton-Raphson\n"
        "  --out DIRECTORY      Reports (default: results; directory must be empty)\n"
        "  --tolerance VALUE    Max P/Q mismatch in pu (default: 1e-8)\n"
        "  --max-iterations N   Iteration limit (default: 100)\n"
        "  --interactive        Prompt for input paths and method\n"
        "  --template DIRECTORY Write editable two-bus input CSVs\n"
        "  --help               Show this help\n\n"
        "All electrical data use a common per-unit base; angles use degrees.\n"
        "Positive P/Q = generation; negative P/Q = consumption.\n"
        "Outputs: input copies, bus/branch/summary/convergence CSV, ASCII network, SVG plots.\n";
}
inline void empty_destination(const std::filesystem::path& dir) {
    if (std::filesystem::exists(dir) && (!std::filesystem::is_directory(dir) || !std::filesystem::is_empty(dir)))
        throw std::runtime_error("Output directory must be empty to preserve existing files: " + dir.string());
}
inline int run(int argc, char** argv) {
    try {
        std::filesystem::path buses, branches, out = "results", templ;
        std::string method = default_method; double tolerance = 1e-8; int iterations = 100;
        bool interactive = false;
        if (argc == 1) { usage(); return 0; }
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") { usage(); return 0; }
            if (key == "--interactive") { interactive = true; continue; }
            if (i + 1 == argc) throw std::runtime_error("Missing value for " + key);
            std::string value = argv[++i];
            if (key == "--buses") buses = value;
            else if (key == "--branches") branches = value;
            else if (key == "--out") out = value;
            else if (key == "--template") templ = value;
            else if (key == "--method") method = value;
            else if (key == "--tolerance") tolerance = number(value);
            else if (key == "--max-iterations") {
                double n = number(value);
                if (n < 1 || n > 100000 || n != std::floor(n)) throw std::runtime_error("Iteration limit must be an integer from 1 to 100000");
                iterations = static_cast<int>(n);
            } else throw std::runtime_error("Unknown option: " + key);
        }
        if (!templ.empty()) {
            if (!buses.empty() || !branches.empty() || interactive) throw std::runtime_error("Use --template separately from a solve");
            empty_destination(templ); std::filesystem::create_directories(templ);
            Case c{{{"1","SLACK",0,0,1,0},{"2","PQ",-.5,-.45,1,0}},{{"L1","1","2",0,.1,0,1,0}}};
            save_input(c, templ); std::cout << "Editable inputs saved in " << templ.string() << '\n'; return 0;
        }
        if (interactive) {
            auto prompt = [](const char* message) { std::string value; std::cout << message; if (!std::getline(std::cin, value)) throw std::runtime_error("Input closed"); return trim(value); };
            buses = prompt("Bus CSV path: "); branches = prompt("Branch CSV path: ");
            auto selected = prompt("Method (gs/nr, Enter for branch default): "); if (!selected.empty()) method = selected;
            auto destination = prompt("Empty report directory (Enter for results): "); if (!destination.empty()) out = destination;
        }
        if (buses.empty() || branches.empty()) throw std::runtime_error("Both --buses and --branches are required; see --help");
        empty_destination(out);
        auto c = load(buses, branches); auto r = solve(c, method, tolerance, iterations);
        report(c, r, method, out);
        std::cout << ascii(c, &r) << "\nMethod: " << method << ", iterations: " << r.mismatch.size() - 1
                  << ", maximum mismatch: " << r.mismatch.back() << " pu\nReports: " << std::filesystem::absolute(out).string() << '\n';
        return r.converged ? 0 : 2;
    } catch (const std::exception& error) { std::cerr << "Error: " << error.what() << '\n'; return 1; }
}
} // namespace lf
