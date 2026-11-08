#pragma once
#include "solver.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace lf {
using Rows = std::vector<std::vector<std::string>>;
inline std::string trim(std::string s) {
    auto first = s.find_first_not_of(" \t\r\n");
    return first == std::string::npos ? "" : s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
inline Rows csv_read(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot read " + path.string());
    std::string data((std::istreambuf_iterator<char>(in)), {});
    if (data.substr(0, 3) == "\xef\xbb\xbf") data.erase(0, 3);
    Rows rows; std::vector<std::string> row; std::string field;
    bool quoted = false, closed = false;
    auto end_field = [&] { row.push_back(trim(field)); field.clear(); closed = false; };
    auto end_row = [&] { end_field(); if (row.size() != 1 || !row[0].empty()) rows.push_back(row); row.clear(); };
    for (size_t i = 0; i < data.size(); ++i) {
        char ch = data[i];
        if (quoted) {
            if (ch == '"') {
                if (i + 1 < data.size() && data[i + 1] == '"') { field += '"'; ++i; }
                else { quoted = false; closed = true; }
            } else field += ch;
        } else if (ch == ',') end_field();
        else if (ch == '\n' || ch == '\r') { if (ch == '\r' && i + 1 < data.size() && data[i + 1] == '\n') ++i; end_row(); }
        else if (ch == '"') { if (!field.empty() || closed) throw std::runtime_error("Malformed CSV quote in " + path.string()); quoted = true; }
        else { if (closed && ch != ' ' && ch != '\t') throw std::runtime_error("Text after CSV quote in " + path.string()); field += ch; }
    }
    if (quoted) throw std::runtime_error("Unterminated CSV quote in " + path.string());
    if (!field.empty() || !row.empty() || closed) end_row();
    if (rows.empty()) throw std::runtime_error("Empty CSV: " + path.string());
    return rows;
}
inline void header(const Rows& rows, const std::vector<std::string>& expected) {
    if (rows[0] != expected) throw std::runtime_error("CSV header does not match documented schema");
    for (size_t i = 1; i < rows.size(); ++i)
        if (rows[i].size() != expected.size()) throw std::runtime_error("Wrong CSV column count at record " + std::to_string(i + 1));
}
inline double number(const std::string& s, bool limit = false) {
    if (limit && s.empty()) return inf;
    size_t used = 0; double d;
    try { d = std::stod(s, &used); } catch (...) { throw std::runtime_error("Invalid number: " + s); }
    if (used != s.size() || !std::isfinite(d)) throw std::runtime_error("Expected finite number: " + s);
    return d;
}
inline const std::vector<std::string> bus_header{"id","type","p_pu","q_pu","vm_pu","va_deg","qmin_pu","qmax_pu"};
inline const std::vector<std::string> branch_header{"id","from","to","r_pu","x_pu","b_pu","tap","shift_deg"};
inline Case load(const std::filesystem::path& buses, const std::filesystem::path& branches) {
    auto u = csv_read(buses), e = csv_read(branches); header(u, bus_header); header(e, branch_header); Case c;
    for (size_t i = 1; i < u.size(); ++i) {
        const auto& r = u[i];
        c.buses.push_back({r[0],r[1],number(r[2]),number(r[3]),number(r[4]),number(r[5]),r[6].empty() ? -inf : number(r[6]),r[7].empty() ? inf : number(r[7])});
    }
    for (size_t i = 1; i < e.size(); ++i) {
        const auto& r = e[i]; c.branches.push_back({r[0],r[1],r[2],number(r[3]),number(r[4]),number(r[5]),number(r[6]),number(r[7])});
    }
    validate(c); return c;
}
inline std::string cell(const std::string& value) {
    std::string out = "\"";
    for (char ch : value) { if (ch == '"') out += '"'; out += ch; }
    return out + '"';
}
inline std::ofstream output(const std::filesystem::path& path) {
    std::ofstream out(path); if (!out) throw std::runtime_error("Cannot write " + path.string());
    out.exceptions(std::ios::badbit | std::ios::failbit); out << std::setprecision(15); return out;
}
inline void save_input(const Case& c, const std::filesystem::path& dir) {
    auto u = output(dir / "input_buses.csv"), e = output(dir / "input_branches.csv");
    u << "id,type,p_pu,q_pu,vm_pu,va_deg,qmin_pu,qmax_pu\n";
    for (const auto& b : c.buses) {
        u << cell(b.id) << ',' << b.type << ',' << b.p << ',' << b.q << ',' << b.vm << ',' << b.va << ',';
        if (std::isfinite(b.qmin)) u << b.qmin;
        u << ',';
        if (std::isfinite(b.qmax)) u << b.qmax;
        u << '\n';
    }
    e << "id,from,to,r_pu,x_pu,b_pu,tap,shift_deg\n";
    for (const auto& b : c.branches) e << cell(b.id) << ',' << cell(b.from) << ',' << cell(b.to) << ',' << b.r << ',' << b.x << ',' << b.b << ',' << b.tap << ',' << b.shift << '\n';
}
inline std::string ascii(const Case& c, const Result* result = nullptr) {
    std::ostringstream out;
    std::vector<C> actual;
    if (result) actual = injections(ybus(c), result->voltage);
    out << "LOAD FLOW NETWORK\n";
    if (result) out << (result->converged ? "CONVERGED" : "NOT CONVERGED - last iterate only") << "\n";
    out << "Each branch is listed once; repeated bus labels denote the same node.\n\n";
    for (const auto& e : c.branches)
        out << '[' << e.from << "]----{" << e.id << "}----[" << e.to << "]  Z=" << e.r << "+j(" << e.x << ") pu  B=" << e.b << " pu  tap=" << e.tap << " shift=" << e.shift << " deg\n";
    out << "\nBUSES (net injections: generation positive, demand negative)\n";
    for (size_t i = 0; i < c.buses.size(); ++i) {
        const auto& b = c.buses[i]; out << '[' << b.id << "] " << b.type;
        if (result) out << " -> " << result->types[i] << "  V=" << std::abs(result->voltage[i]) << " pu / " << std::arg(result->voltage[i]) * 180 / pi << " deg";
        out << "  P=" << (result ? actual[i].real() : b.p)
            << " Q=" << (result ? actual[i].imag() : b.q) << " pu\n";
    }
    return out.str();
}
inline std::string xml(const std::string& s) {
    std::string out; for (char ch : s) { if (ch == '&') out += "&amp;"; else if (ch == '<') out += "&lt;"; else if (ch == '>') out += "&gt;"; else if (ch == '"') out += "&quot;"; else out += ch; } return out;
}
inline void plot(const std::filesystem::path& path, const std::string& title,
                 const std::vector<std::string>& labels, const std::vector<double>& values, const std::string& unit) {
    auto out = output(path); double low = 0, high = 0;
    for (double v : values) { low = std::min(low, v); high = std::max(high, v); }
    double span = high - low; if (span < 1e-12) span = 1; low -= span * .1; high += span * .1;
    double width = std::max(800.0, 80.0 + 36.0 * values.size());
    auto x = [&](size_t i) { return 65 + (width - 100) * (i + .5) / std::max(size_t(1), values.size()); };
    auto y = [&](double v) { return 330 - (v - low) / (high - low) * 260; };
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"440\" viewBox=\"0 0 " << width << " 440\"><rect width=\"100%\" height=\"100%\" fill=\"#fff\"/><g font-family=\"sans-serif\" fill=\"#172235\"><text x=\"30\" y=\"30\" font-size=\"20\">" << xml(title) << "</text><text x=\"30\" y=\"52\">" << xml(unit) << "</text>";
    for (int k = 0; k <= 5; ++k) { double v = low + (high - low) * k / 5; out << "<path d=\"M60 " << y(v) << " H" << width - 20 << "\" stroke=\"#e0e5eb\"/><text x=\"3\" y=\"" << y(v) << "\" font-size=\"10\">" << std::setprecision(4) << v << "</text>"; }
    out << "<polyline fill=\"none\" stroke=\"#1769aa\" stroke-width=\"2\" points=\"";
    for (size_t i = 0; i < values.size(); ++i) out << x(i) << ',' << y(values[i]) << ' ';
    out << "\"/>";
    for (size_t i = 0; i < values.size(); ++i) out << "<circle cx=\"" << x(i) << "\" cy=\"" << y(values[i]) << "\" r=\"3\" fill=\"#1769aa\"/><text transform=\"translate(" << x(i) << ",350) rotate(45)\" font-size=\"11\">" << xml(labels[i]) << "</text>";
    out << "</g></svg>\n";
}
inline void report(const Case& c, const Result& r, const std::string& method, const std::filesystem::path& dir) {
    std::filesystem::create_directories(dir); save_input(c, dir);
    auto bus = output(dir / "buses.csv"), branch = output(dir / "branches.csv"), history = output(dir / "convergence.csv");
    auto s = injections(ybus(c), r.voltage); auto ids = indices(c); C losses = 0;
    bus << "id,requested_type,solved_type,vm_pu,va_deg,p_injection_pu,q_injection_pu,converged\n";
    std::vector<std::string> labels; std::vector<double> vm, va;
    for (size_t i = 0; i < c.buses.size(); ++i) {
        labels.push_back(c.buses[i].id); vm.push_back(std::abs(r.voltage[i])); va.push_back(std::arg(r.voltage[i]) * 180 / pi);
        bus << cell(c.buses[i].id) << ',' << c.buses[i].type << ',' << r.types[i] << ',' << vm.back() << ',' << va.back() << ',' << s[i].real() << ',' << s[i].imag() << ',' << r.converged << '\n';
    }
    branch << "id,from,to,p_from_pu,q_from_pu,p_to_pu,q_to_pu,p_loss_pu,q_loss_pu,converged\n";
    for (const auto& e : c.branches) {
        auto a = ids.at(e.from), b = ids.at(e.to); auto st = stamp(e);
        C sf = r.voltage[a] * std::conj(st.ff * r.voltage[a] + st.ft * r.voltage[b]);
        C stp = r.voltage[b] * std::conj(st.tf * r.voltage[a] + st.tt * r.voltage[b]); C loss = sf + stp; losses += loss;
        branch << cell(e.id) << ',' << cell(e.from) << ',' << cell(e.to) << ',' << sf.real() << ',' << sf.imag() << ',' << stp.real() << ',' << stp.imag() << ',' << loss.real() << ',' << loss.imag() << ',' << r.converged << '\n';
    }
    history << "iteration,max_mismatch_pu\n"; std::vector<std::string> iterations;
    for (size_t i = 0; i < r.mismatch.size(); ++i) { history << i << ',' << r.mismatch[i] << '\n'; iterations.push_back(std::to_string(i)); }
    auto summary = output(dir / "summary.csv");
    summary << "method,converged,iterations,max_mismatch_pu,p_loss_pu,q_loss_pu\n" << method << ',' << r.converged << ',' << r.mismatch.size() - 1 << ',' << r.mismatch.back() << ',' << losses.real() << ',' << losses.imag() << '\n';
    auto diagram = output(dir / "network.txt"); diagram << ascii(c, &r);
    std::string suffix = r.converged ? "" : " (NOT CONVERGED)";
    plot(dir / "voltage.svg", "Bus voltage magnitude" + suffix, labels, vm, "per unit");
    plot(dir / "angles.svg", "Bus voltage angle" + suffix, labels, va, "degrees");
    plot(dir / "convergence.svg", "Power mismatch" + suffix, iterations, r.mismatch, "maximum absolute P/Q mismatch, per unit");
}
} // namespace lf
