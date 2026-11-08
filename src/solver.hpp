#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace lf {
using C = std::complex<double>;
using Matrix = std::vector<std::vector<C>>;
constexpr double pi = 3.14159265358979323846;
constexpr double inf = std::numeric_limits<double>::infinity();
struct Bus {
    std::string id, type;
    double p = 0, q = 0, vm = 1, va = 0, qmin = -inf, qmax = inf;
};
struct Branch {
    std::string id, from, to;
    double r = 0, x = 0, b = 0, tap = 1, shift = 0;
};
struct Case { std::vector<Bus> buses; std::vector<Branch> branches; };
struct Result {
    std::vector<C> voltage;
    std::vector<std::string> types;
    std::vector<double> mismatch;
    bool converged = false;
};
inline std::unordered_map<std::string, size_t> indices(const Case& c) {
    std::unordered_map<std::string, size_t> ids;
    for (size_t i = 0; i < c.buses.size(); ++i)
        if (c.buses[i].id.empty() || !ids.emplace(c.buses[i].id, i).second)
            throw std::runtime_error("Bus IDs must be nonempty and unique");
    return ids;
}
inline void validate(const Case& c) {
    if (c.buses.empty() || c.buses.size() > 500)
        throw std::runtime_error("Provide 1 to 500 buses (dense educational solver)");
    auto ids = indices(c);
    size_t slack = 0, root = 0;
    for (size_t i = 0; i < c.buses.size(); ++i) {
        const auto& u = c.buses[i];
        if (u.type == "SLACK") { ++slack; root = i; }
        else if (u.type != "PQ" && u.type != "PV") throw std::runtime_error("Bus type must be SLACK, PQ or PV");
        if (!std::isfinite(u.p) || !std::isfinite(u.q) || !std::isfinite(u.vm) ||
            !std::isfinite(u.va) || u.vm <= 0 || std::isnan(u.qmin) || std::isnan(u.qmax) ||
            u.qmin > u.qmax || u.qmin == inf || u.qmax == -inf)
            throw std::runtime_error("Invalid bus data at " + u.id);
    }
    if (slack != 1) throw std::runtime_error("Exactly one SLACK bus is required");
    std::vector<std::vector<size_t>> edges(c.buses.size());
    std::unordered_map<std::string, bool> branch_ids;
    for (const auto& e : c.branches) {
        if (e.id.empty() || !branch_ids.emplace(e.id, true).second || !ids.count(e.from) ||
            !ids.count(e.to) || e.from == e.to || !std::isfinite(e.r) || !std::isfinite(e.x) ||
            !std::isfinite(e.b) || !std::isfinite(e.tap) || !std::isfinite(e.shift) ||
            e.r < 0 || std::abs(C(e.r, e.x)) < 1e-12 || e.tap <= 0)
            throw std::runtime_error("Invalid branch data at " + e.id);
        auto a = ids.at(e.from), b = ids.at(e.to);
        edges[a].push_back(b); edges[b].push_back(a);
    }
    std::vector<size_t> todo{root}; std::vector<bool> seen(c.buses.size()); seen[root] = true;
    for (size_t k = 0; k < todo.size(); ++k)
        for (auto v : edges[todo[k]]) if (!seen[v]) { seen[v] = true; todo.push_back(v); }
    if (todo.size() != c.buses.size()) throw std::runtime_error("Network is disconnected from the SLACK bus");
}
struct Stamp { C ff, ft, tf, tt; };
inline Stamp stamp(const Branch& e) {
    C y = 1.0 / C(e.r, e.x), charging(0, e.b / 2), t = std::polar(e.tap, e.shift * pi / 180);
    return {(y + charging) / std::norm(t), -y / std::conj(t), -y / t, y + charging};
}
inline Matrix ybus(const Case& c) {
    validate(c);
    auto ids = indices(c); Matrix y(c.buses.size(), std::vector<C>(c.buses.size()));
    for (const auto& e : c.branches) {
        auto a = ids.at(e.from), b = ids.at(e.to); auto s = stamp(e);
        y[a][a] += s.ff; y[a][b] += s.ft; y[b][a] += s.tf; y[b][b] += s.tt;
    }
    for (size_t i = 0; i < y.size(); ++i)
        if (y.size() > 1 && std::abs(y[i][i]) < 1e-12) throw std::runtime_error("Zero Y-bus diagonal");
    return y;
}
inline std::vector<C> injections(const Matrix& y, const std::vector<C>& v) {
    std::vector<C> s(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        C current = 0; for (size_t j = 0; j < v.size(); ++j) current += y[i][j] * v[j];
        s[i] = v[i] * std::conj(current);
    }
    return s;
}
inline double norm(const std::vector<double>& v) {
    double n = 0;
    for (double x : v) { if (!std::isfinite(x)) throw std::runtime_error("Non-finite numerical result"); n = std::max(n, std::abs(x)); }
    return n;
}
inline std::vector<double> linear_solve(std::vector<std::vector<double>> a, std::vector<double> b) {
    const size_t n = b.size();
    for (size_t k = 0; k < n; ++k) {
        size_t pivot = k;
        for (size_t i = k + 1; i < n; ++i) if (std::abs(a[i][k]) > std::abs(a[pivot][k])) pivot = i;
        if (std::abs(a[pivot][k]) < 1e-12) throw std::runtime_error("Singular Jacobian; check operating point and network");
        std::swap(a[k], a[pivot]); std::swap(b[k], b[pivot]);
        for (size_t i = k + 1; i < n; ++i) {
            double f = a[i][k] / a[k][k];
            for (size_t j = k + 1; j < n; ++j) a[i][j] -= f * a[k][j];
            b[i] -= f * b[k];
        }
    }
    std::vector<double> x(n);
    for (size_t i = n; i-- > 0;) {
        double z = b[i]; for (size_t j = i + 1; j < n; ++j) z -= a[i][j] * x[j];
        x[i] = z / a[i][i];
    }
    return x;
}
inline Result solve(const Case& c, const std::string& method = "nr", double tolerance = 1e-8, int max_iterations = 100) {
    if ((method != "nr" && method != "gs") || !std::isfinite(tolerance) || tolerance <= 0 || max_iterations < 1)
        throw std::runtime_error("Invalid method, tolerance or iteration limit");
    auto y = ybus(c); const size_t n = c.buses.size(); Result out;
    std::vector<double> q(n);
    for (const auto& b : c.buses) { out.voltage.push_back(std::polar(b.vm, b.va * pi / 180)); out.types.push_back(b.type); }
    for (size_t i = 0; i < n; ++i) q[i] = c.buses[i].q;
    for (int step = 0; step <= max_iterations; ++step) {
        auto s = injections(y, out.voltage);
        // Enforce limits only after solving the current bus-type system. A clamped PV
        // remains PQ for the rest of this run; limits apply to NET bus Q injection.
        auto mismatch_for = [&](const std::vector<C>& values) {
            auto powers = injections(y, values); std::vector<double> f;
            for (size_t i = 0; i < n; ++i) if (out.types[i] != "SLACK") f.push_back(c.buses[i].p - powers[i].real());
            for (size_t i = 0; i < n; ++i) if (out.types[i] == "PQ") f.push_back(q[i] - powers[i].imag());
            return f;
        };
        auto f = mismatch_for(out.voltage);
        double error = norm(f);
        if (error < tolerance) {
            bool changed = false;
            for (size_t i = 0; i < n; ++i) if (out.types[i] == "PV") {
                double bounded = std::clamp(s[i].imag(), c.buses[i].qmin, c.buses[i].qmax);
                if (std::abs(bounded - s[i].imag()) > tolerance) { q[i] = bounded; out.types[i] = "PQ"; changed = true; }
            }
            if (changed) { f = mismatch_for(out.voltage); error = norm(f); }
            else { out.mismatch.push_back(error); out.converged = true; return out; }
        }
        out.mismatch.push_back(error);
        if (step == max_iterations) break;
        if (method == "gs") {
            for (size_t i = 0; i < n; ++i) if (out.types[i] != "SLACK") {
                C sum = 0; for (size_t j = 0; j < n; ++j) if (i != j) sum += y[i][j] * out.voltage[j];
                double qi = q[i];
                if (out.types[i] == "PV") qi = (out.voltage[i] * std::conj(sum + y[i][i] * out.voltage[i])).imag();
                C next = (std::conj(C(c.buses[i].p, qi) / out.voltage[i]) - sum) / y[i][i];
                if (std::abs(next) < 1e-10 || !std::isfinite(std::abs(next))) throw std::runtime_error("Voltage collapse or numerical failure");
                out.voltage[i] = out.types[i] == "PV" ? std::polar(c.buses[i].vm, std::arg(next)) : next;
            }
        } else {
            std::vector<size_t> angles, pq;
            for (size_t i = 0; i < n; ++i) { if (out.types[i] != "SLACK") angles.push_back(i); if (out.types[i] == "PQ") pq.push_back(i); }
            size_t na = angles.size(), m = na + pq.size();
            std::vector<std::vector<double>> jac(m, std::vector<double>(m));
            for (size_t row = 0; row < m; ++row) {
                bool reactive = row >= na; size_t i = reactive ? pq[row - na] : angles[row];
                double vi = std::abs(out.voltage[i]);
                for (size_t col = 0; col < m; ++col) {
                    bool magnitude = col >= na; size_t j = magnitude ? pq[col - na] : angles[col];
                    double vj = std::abs(out.voltage[j]), d = std::arg(out.voltage[i]) - std::arg(out.voltage[j]);
                    double g = y[i][j].real(), b = y[i][j].imag();
                    if (!reactive && !magnitude) jac[row][col] = i == j ? -s[i].imag() - b * vi * vi : vi * vj * (g * std::sin(d) - b * std::cos(d));
                    if (!reactive && magnitude) jac[row][col] = i == j ? s[i].real() / vi + g * vi : vi * (g * std::cos(d) + b * std::sin(d));
                    if (reactive && !magnitude) jac[row][col] = i == j ? s[i].real() - g * vi * vi : -vi * vj * (g * std::cos(d) + b * std::sin(d));
                    if (reactive && magnitude) jac[row][col] = i == j ? s[i].imag() / vi - b * vi : vi * (g * std::sin(d) - b * std::cos(d));
                }
            }
            auto dx = linear_solve(jac, f); bool accepted = false;
            for (double alpha = 1; alpha >= 1.0 / 1024; alpha /= 2) {
                auto trial = out.voltage; std::vector<double> vm(n), va(n);
                for (size_t i = 0; i < n; ++i) { vm[i] = std::abs(trial[i]); va[i] = std::arg(trial[i]); }
                for (size_t k = 0; k < na; ++k) va[angles[k]] += alpha * dx[k];
                for (size_t k = 0; k < pq.size(); ++k) vm[pq[k]] += alpha * dx[na + k];
                bool positive = true;
                for (size_t i = 0; i < n; ++i) { positive = positive && vm[i] > 1e-8 && std::isfinite(vm[i]) && std::isfinite(va[i]); trial[i] = std::polar(vm[i], va[i]); }
                if (positive && norm(mismatch_for(trial)) < error) { out.voltage = trial; accepted = true; break; }
            }
            if (!accepted) return out;
        }
    }
    return out;
}
} // namespace lf
