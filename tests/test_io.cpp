#include "../src/app.hpp"
#include <chrono>
#include <iostream>

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
    bool rejected = false; try { f(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "Invalid input accepted");
}
int main() {
    // Own only this newly created directory; never reuse or delete a pre-existing path.
    auto dir = std::filesystem::temp_directory_path() / ("lf-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(dir)) return 1;
    try {
        lf::Case c{{{"Source, \"A\"","SLACK",0,0,1,0},{"Load<&>","PQ",-.5,-.45,1,0}},
                   {{"Line,1","Source, \"A\"","Load<&>",0,.1,0,1,0}}};
        lf::save_input(c, dir);
        auto roundtrip = lf::load(dir / "input_buses.csv", dir / "input_branches.csv");
        require(roundtrip.buses[0].id == c.buses[0].id, "Quoted ID roundtrip failed");
        require(roundtrip.branches[0].id == c.branches[0].id, "Branch ID roundtrip failed");
        require(roundtrip.buses[0].qmin == -lf::inf, "Blank Q limit roundtrip failed");
        auto r = lf::solve(roundtrip, "nr"); require(r.converged, "Roundtrip case did not solve");
        lf::report(roundtrip, r, "nr", dir / "reports");
        auto rows = lf::csv_read(dir / "reports" / "buses.csv");
        require(rows.size() == 3 && rows[2][7] == "1", "Bus output rows/status wrong");
        require(std::abs(lf::number(rows[2][3]) - std::abs(r.voltage[1])) < 1e-12, "Output precision wrong");
        auto flows = lf::csv_read(dir / "reports" / "branches.csv");
        require(std::abs(lf::number(flows[1][3]) + lf::number(flows[1][5])) < 1e-10, "Lossless branch P balance wrong");
        auto svg = dir / "reports" / "voltage.svg";
        std::ifstream in(svg); std::string content((std::istreambuf_iterator<char>(in)), {});
        require(content.find("Load&lt;&amp;&gt;") != std::string::npos, "SVG label not escaped");
        in.close();
        require(lf::ascii(c, &r).find("Line,1") != std::string::npos, "ASCII missing branch");
        rejects([&]{ lf::empty_destination(dir); });
        rejects([]{ lf::number("nan"); }); rejects([]{ lf::number("1oops"); });
        auto malformed = dir / "malformed.csv";
        { auto out = lf::output(malformed); out << "a,b\n\"unclosed"; }
        rejects([&]{ lf::csv_read(malformed); });
        { auto out = lf::output(malformed); out << "a,b\n\"x\"oops,b\n"; }
        rejects([&]{ lf::csv_read(malformed); });
        { auto out = lf::output(malformed); out << "\xef\xbb\xbf" << "a,b\r\n\"x,y\",z\r\n"; }
        require(lf::csv_read(malformed)[1][0] == "x,y", "BOM/CRLF failed");
        auto failed = lf::solve(c, "nr", 1e-14, 1); require(!failed.converged, "Expected nonconvergence");
        lf::report(c, failed, "nr", dir / "failed");
        require(lf::csv_read(dir / "failed" / "summary.csv")[1][1] == "0", "Nonconvergence not exported");
        std::filesystem::remove_all(dir);
        std::cout << "PASS CSV roundtrip, malformed inputs, report balance, SVG escaping, nonconvergence\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\nTest evidence retained at " << dir.string() << '\n'; return 1;
    }
}
