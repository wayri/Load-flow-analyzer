#include "../src/solver.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace lf;
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance, const std::string& label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
        throw std::runtime_error(label + ": expected " + std::to_string(expected)
                                 + ", got " + std::to_string(actual));
}
template <class F> void rejects(F action, const std::string& label) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, label + " was accepted");
}
Bus bus(std::string id, std::string type, double p, double q, double vm = 1.0,
        double va = 0.0) {
    Bus value;
    value.id = id; value.type = type; value.p = p; value.q = q;
    value.vm = vm; value.va = va;
    return value;
}
Branch line(std::string id, std::string from, std::string to, double r, double x,
            double b = 0.0) {
    Branch value;
    value.id = id; value.from = from; value.to = to;
    value.r = r; value.x = x; value.b = b; value.tap = 1.0; value.shift = 0.0;
    return value;
}
Case two_bus() {
    return {{bus("source", "SLACK", 0, 0), bus("load", "PQ", -0.5, -0.45)},
            {line("L1", "source", "load", 0.0, 0.1)}};
}
void exact_two_bus() {
    // With V1=1, X=.1 and V2=.95-j.05, V2*conj((V2-V1)/(jX))=-.5-j.45.
    // This analytic reference is independent of either iterative algorithm.
    for (const auto& method : {"nr", "gs"}) {
        auto c = two_bus();
        auto result = solve(c, method, 1e-10, 1000);
        require(result.converged, std::string(method) + " two-bus convergence");
        near(result.voltage[1].real(), 0.95, 1e-8, "two-bus real voltage");
        near(result.voltage[1].imag(), -0.05, 1e-8, "two-bus imaginary voltage");
        auto power = injections(ybus(c), result.voltage);
        near(power[0].real() + power[1].real(), 0, 1e-9, "lossless active balance");
        near(power[1].real(), -0.5, 1e-9, "PQ active injection");
        near(power[1].imag(), -0.45, 1e-9, "PQ reactive injection");
        c.buses[0].va = 23.0; c.buses[1].va = 23.0;
        auto rotated = solve(c, method, 1e-10, 1000);
        require(rotated.converged, "rotated slack reference convergence");
        const auto reference = std::complex<double>(.95, -.05)
            * std::polar(1.0, 23.0 * std::acos(-1.0) / 180.0);
        near(std::abs(rotated.voltage[1] - reference), 0, 1e-8, "slack reference rotation");
    }
}
void pv_and_limits() {
    for (const auto& method : {"nr", "gs"}) {
        Case c{{bus("1", "SLACK", 0, 0), bus("2", "PV", 0.4, 0, 1.02)},
               {line("L1", "1", "2", 0, 0.1)}};
        auto free = solve(c, method, 1e-10, 1000);
        require(free.converged, "PV convergence");
        near(std::abs(free.voltage[1]), 1.02, 1e-9, "PV voltage setpoint");
        near(free.voltage[1].imag(), .04, 1e-8, "PV active power angle");
        const double expected_q = (1.02 * 1.02 - std::sqrt(1.02 * 1.02 - .04 * .04)) / .1;
        near(injections(ybus(c), free.voltage)[1].imag(), expected_q, 1e-8, "PV reactive injection");
        c.buses[1].qmin = .21; c.buses[1].qmax = .22;
        auto feasible = solve(c, method, 1e-10, 1000);
        require(feasible.converged && feasible.types[1] == "PV", "feasible PV must not clamp at flat start");
        for (bool upper : {true, false}) {
            c.buses[1].qmin = upper ? -1.0 : .3;
            c.buses[1].qmax = upper ? .1 : 1.0;
            auto limited = solve(c, method, 1e-10, 1000);
            require(limited.converged, "Q-limited convergence");
            require(limited.types[1] == "PQ", "Q-limit changes PV to PQ");
            auto power = injections(ybus(c), limited.voltage);
            near(power[1].real(), .4, 1e-8, "Q-limit preserves P");
            near(power[1].imag(), upper ? .1 : .3, 1e-8, "reactive limit");
        }
    }
}
Case nine_bus() {
    // Network and injections: official MATPOWER case9.m, baseMVA=100.
    // https://github.com/MATPOWER/matpower/blob/master/data/case9.m
    // Generator voltage setpoints (runpf behavior): 1.04, 1.025, 1.025.
    // The test checks equations, conservation and independent algorithms;
    // it does not claim an externally executed MATPOWER comparison.
    return {{bus("1", "SLACK", 0, 0, 1.04), bus("2", "PV", 1.63, 0, 1.025),
             bus("3", "PV", .85, 0, 1.025), bus("4", "PQ", 0, 0),
             bus("5", "PQ", -.9, -.3), bus("6", "PQ", 0, 0),
             bus("7", "PQ", -1, -.35), bus("8", "PQ", 0, 0),
             bus("9", "PQ", -1.25, -.5)},
            {line("1", "1", "4", 0, .0576), line("2", "4", "5", .017, .092, .158),
             line("3", "5", "6", .039, .17, .358), line("4", "3", "6", 0, .0586),
             line("5", "6", "7", .0119, .1008, .209),
             line("6", "7", "8", .0085, .072, .149), line("7", "8", "2", 0, .0625),
             line("8", "8", "9", .032, .161, .306), line("9", "9", "4", .01, .085, .176)}};
}
void meshed_network() {
    const auto c = nine_bus();
    const auto nr = solve(c, "nr", 1e-10, 100);
    const auto gs = solve(c, "gs", 1e-10, 3000);
    require(nr.converged && gs.converged, "nine-bus convergence");
    const auto power = injections(ybus(c), nr.voltage);
    for (std::size_t i = 0; i < c.buses.size(); ++i) {
        near(std::abs(nr.voltage[i] - gs.voltage[i]), 0, 1e-8, "GS/NR nine-bus agreement");
        if (i != 0) near(power[i].real(), c.buses[i].p, 1e-8, "nine-bus P residual");
        if (c.buses[i].type == "PQ")
            near(power[i].imag(), c.buses[i].q, 1e-8, "nine-bus Q residual");
    }
    double nodal_loss = 0, branch_loss = 0;
    for (auto s : power) nodal_loss += s.real();
    for (const auto& e : c.branches) {
        const int from = std::stoi(e.from) - 1, to = std::stoi(e.to) - 1;
        const auto current = (nr.voltage[from] - nr.voltage[to]) / std::complex<double>(e.r, e.x);
        branch_loss += e.r * std::norm(current);
    }
    near(nodal_loss, branch_loss, 1e-9, "resistive branch/nodal loss balance");
    require(nodal_loss > 0, "positive real losses");
}
void transformer_stamp() {
    auto c = two_bus();
    c.branches[0].tap = 1.1; c.branches[0].shift = 15; c.branches[0].b = .02;
    const auto y = ybus(c);
    const std::complex<double> series(0, -10), charging(0, .01);
    const auto tap = std::polar(1.1, 15.0 * std::acos(-1.0) / 180.0);
    near(std::abs(y[0][0] - (series + charging) / std::norm(tap)), 0, 1e-12, "tap diagonal");
    near(std::abs(y[0][1] + series / std::conj(tap)), 0, 1e-12, "tap forward admittance");
    near(std::abs(y[1][0] + series / tap), 0, 1e-12, "tap reverse admittance");
    near(std::abs(y[1][1] - series - charging), 0, 1e-12, "receiving admittance");
}
void validation_and_failure() {
    auto c = two_bus();
    auto failed = solve(c, "nr", 1e-12, 1);
    require(!failed.converged, "iteration exhaustion must not report convergence");
    rejects([&] { solve(c, "invalid"); }, "unknown method");
    rejects([&] { solve(c, "nr", 0); }, "zero tolerance");
    rejects([&] { solve(c, "nr", 1e-8, 0); }, "zero iterations");
    c.branches.clear();
    rejects([&] { solve(c); }, "disconnected buses");
    c = two_bus(); c.buses[1].id = c.buses[0].id;
    rejects([&] { solve(c); }, "duplicate bus ID");
    c = two_bus(); c.buses[1].type = "SLACK";
    rejects([&] { solve(c); }, "multiple slack buses");
    c = two_bus(); c.branches[0].to = "missing";
    rejects([&] { solve(c); }, "unknown branch endpoint");
    c = two_bus(); c.branches[0].x = 0;
    rejects([&] { solve(c); }, "zero impedance");
    c = two_bus(); c.buses[1].vm = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { solve(c); }, "nonfinite voltage");
    c = two_bus(); c.buses[1].qmin = 2; c.buses[1].qmax = 1;
    rejects([&] { solve(c); }, "reversed Q bounds");
}
}
int main() {
    int failures = 0;
    const std::pair<const char*, std::function<void()>> tests[] = {
        {"analytic two-bus reference", exact_two_bus}, {"PV and reactive limits", pv_and_limits},
        {"MATPOWER nine-bus network", meshed_network}, {"transformer Ybus stamp", transformer_stamp},
        {"invalid data and nonconvergence", validation_and_failure}};
    for (const auto& test : tests) {
        try { test.second(); std::cout << "PASS " << test.first << '\n'; }
        catch (const std::exception& e) { ++failures; std::cerr << "FAIL " << test.first << ": " << e.what() << '\n'; }
    }
    return failures ? 1 : 0;
}
