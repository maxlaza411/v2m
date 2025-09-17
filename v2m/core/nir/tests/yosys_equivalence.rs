use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use serde_json::Value;
use tempfile::{Builder, TempDir};

use v2m_formats::nir::{
    BitRef, BitRefConcat, BitRefConst, BitRefNet, Module, Net, Nir, Node, NodeOp, Port,
    PortDirection,
};
use v2m_nir::nir_to_verilog;

const FA8_SPEC: &str = r#"module fa8_spec(
    input wire [7:0] a,
    input wire [7:0] b,
    input wire cin,
    output wire [7:0] sum,
    output wire cout
);
    wire [8:0] total;
    assign total = {1'b0, a} + {b, cin};
    assign sum = total[7:0];
    assign cout = total[8];
endmodule
"#;

const ALU32_SPEC: &str = r#"module alu32_spec(
    input wire [31:0] a,
    input wire [31:0] b,
    input wire [1:0] op,
    output wire [31:0] y
);
    wire [31:0] sum = a + b;
    wire [31:0] and_ab = a & b;
    wire [31:0] or_ab = a | b;
    wire [31:0] xor_ab = a ^ b;
    wire [31:0] lo_sel = op[0] ? and_ab : sum;
    wire [31:0] hi_sel = op[0] ? xor_ab : or_ab;
    assign y = op[1] ? hi_sel : lo_sel;
endmodule
"#;

const COUNTER_SPEC: &str = r#"module counter4_spec(
    input wire clk,
    input wire rst,
    output reg [3:0] count
);
    always @(posedge clk) begin
        if (rst) begin
            count <= 4'd0;
        end else begin
            count <= count + 4'd1;
        end
    end
endmodule
"#;

const FIFO_SPEC: &str = r#"module fifo_small_spec(
    input wire clk,
    input wire rst,
    input wire enq_valid,
    output wire enq_ready,
    input wire [7:0] enq_data,
    input wire deq_ready,
    output wire deq_valid,
    output wire [7:0] deq_data
);
    reg valid;
    reg [7:0] data;

    assign enq_ready = ~valid | deq_ready;
    assign deq_valid = valid;
    assign deq_data = data;

    wire hold_valid = valid & ~deq_ready;
    wire enq_fire = enq_valid & enq_ready;
    wire next_valid = hold_valid | enq_fire;
    wire [7:0] data_after_enq = enq_fire ? enq_data : data;

    always @(posedge clk) begin
        if (rst) begin
            valid <= 1'b0;
            data <= 8'h00;
        end else begin
            valid <= next_valid;
            data <= data_after_enq;
        end
    end
endmodule
"#;

#[test]
fn yosys_equivalence_fa8() {
    let design = build_fa8_nir();
    check_comb_equivalence(&design, FA8_SPEC, "fa8_spec");
}

#[test]
fn yosys_equivalence_alu32() {
    let design = build_alu32_nir();
    check_comb_equivalence(&design, ALU32_SPEC, "alu32_spec");
}

#[test]
fn yosys_equivalence_counter() {
    let design = build_counter_nir();
    check_sequential_equivalence(&design, COUNTER_SPEC, "counter4_spec", 8);
}

#[test]
fn yosys_equivalence_fifo() {
    let design = build_fifo_nir();
    check_sequential_equivalence(&design, FIFO_SPEC, "fifo_small_spec", 8);
}

fn workspace_tempdir() -> TempDir {
    static BASE: OnceLock<PathBuf> = OnceLock::new();
    let base = BASE.get_or_init(|| {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir
            .parent()
            .and_then(|dir| dir.parent())
            .expect("workspace root");
        let path = workspace_root.join("target/yosys-tmp");
        fs::create_dir_all(&path).expect("create yosys temp root");
        path
    });
    Builder::new()
        .prefix("yosys-test-")
        .tempdir_in(base)
        .expect("create tempdir in workspace")
}

fn check_comb_equivalence(design: &Nir, spec: &str, spec_top: &str) {
    let dir = workspace_tempdir();
    let dut_path = write_temp(
        dir.path(),
        "dut.v",
        &nir_to_verilog(design).expect("export verilog"),
    );
    let spec_path = write_temp(dir.path(), "spec.v", spec);
    let script_path = write_temp(
        dir.path(),
        "check.ys",
        &comb_script(&spec_path, spec_top, &dut_path, design.top.as_str()),
    );
    run_yosys(&script_path);
}

fn check_sequential_equivalence(design: &Nir, spec: &str, spec_top: &str, cycles: u32) {
    let dir = workspace_tempdir();
    let dut_path = write_temp(
        dir.path(),
        "dut.v",
        &nir_to_verilog(design).expect("export verilog"),
    );
    let spec_path = write_temp(dir.path(), "spec.v", spec);
    let script_path = write_temp(
        dir.path(),
        "check.ys",
        &seq_script(&spec_path, spec_top, &dut_path, design.top.as_str(), cycles),
    );
    run_yosys(&script_path);
}

#[derive(Clone)]
struct YosysCommand {
    program: PathBuf,
    args: Vec<OsString>,
}

fn ensure_yosys() -> YosysCommand {
    static COMMAND: OnceLock<YosysCommand> = OnceLock::new();
    COMMAND
        .get_or_init(|| locate_yosys().unwrap_or_else(install_yowasp_yosys))
        .clone()
}

fn locate_yosys() -> Option<YosysCommand> {
    if let Some(explicit) = std::env::var_os("YOSYS_CMD") {
        let path = PathBuf::from(&explicit);
        if path.as_os_str().is_empty() {
            panic!("YOSYS_CMD environment variable was set but empty");
        }
        let output = Command::new(&path)
            .arg("-V")
            .output()
            .unwrap_or_else(|err| panic!("failed to invoke YOSYS_CMD {}: {}", path.display(), err));
        if !output.status.success() {
            panic!(
                "YOSYS_CMD {} failed version probe with status {}:\nstdout:\n{}\nstderr:\n{}",
                path.display(),
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        return Some(YosysCommand {
            program: path,
            args: Vec::new(),
        });
    }

    match Command::new("yosys").arg("-V").output() {
        Ok(output) => {
            if output.status.success() {
                Some(YosysCommand {
                    program: PathBuf::from("yosys"),
                    args: Vec::new(),
                })
            } else {
                panic!(
                    "found yosys in PATH but version probe failed with status {}:\nstdout:\n{}\nstderr:\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
        Err(err) if err.kind() == ErrorKind::NotFound => None,
        Err(err) => panic!("failed to probe yosys in PATH: {}", err),
    }
}

const YOWASP_PIP_PACKAGE: &str = "yowasp-yosys==0.57.0.0.post986";
const YOWASP_LAUNCHER: &str =
    "import sys, yowasp_yosys; sys.exit(yowasp_yosys.run_yosys(sys.argv[1:]))";

fn install_yowasp_yosys() -> YosysCommand {
    let python = python_command();

    match Command::new(&python)
        .arg("-c")
        .arg("import yowasp_yosys")
        .output()
    {
        Ok(output) if output.status.success() => return yowasp_command(python),
        Ok(_) => {}
        Err(err) if err.kind() == ErrorKind::NotFound => {
            panic!(
                "python interpreter `{}` not found while attempting to install Yosys",
                python.display()
            );
        }
        Err(err) => {
            panic!(
                "failed to invoke `{}` for yowasp-yosys availability check: {}",
                python.display(),
                err
            );
        }
    }

    let pip_output = Command::new(&python)
        .args([
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            YOWASP_PIP_PACKAGE,
        ])
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "failed to invoke `{}` to install {YOWASP_PIP_PACKAGE}: {}",
                python.display(),
                err
            )
        });

    if !pip_output.status.success() {
        panic!(
            "pip failed to install {YOWASP_PIP_PACKAGE} (status {}):\nstdout:\n{}\nstderr:\n{}\nInstall yosys manually and make it available via PATH or YOSYS_CMD.",
            pip_output.status,
            String::from_utf8_lossy(&pip_output.stdout),
            String::from_utf8_lossy(&pip_output.stderr)
        );
    }

    let verify_output = Command::new(&python)
        .arg("-c")
        .arg("import yowasp_yosys")
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "failed to invoke `{}` after installing yowasp-yosys: {}",
                python.display(),
                err
            )
        });

    if !verify_output.status.success() {
        panic!(
            "python `{}` could not import yowasp_yosys even after installation:\nstdout:\n{}\nstderr:\n{}",
            python.display(),
            String::from_utf8_lossy(&verify_output.stdout),
            String::from_utf8_lossy(&verify_output.stderr)
        );
    }

    yowasp_command(python)
}

fn python_command() -> PathBuf {
    std::env::var_os("PYTHON3")
        .or_else(|| std::env::var_os("PYTHON"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("python3"))
}

fn yowasp_command(python: PathBuf) -> YosysCommand {
    YosysCommand {
        program: python,
        args: vec![OsString::from("-c"), OsString::from(YOWASP_LAUNCHER)],
    }
}

fn run_yosys(script: &Path) {
    let command = ensure_yosys();
    let mut process = Command::new(&command.program);
    process.args(&command.args);
    let output = process
        .arg("-q")
        .arg("-s")
        .arg(script)
        .output()
        .expect("invoke yosys");
    if !output.status.success() {
        panic!(
            "yosys failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn comb_script(spec: &Path, spec_top: &str, dut: &Path, dut_top: &str) -> String {
    format!(
        "read_verilog \"{}\"\n\
         prep -top {}\n\
         rename {} spec\n\
         design -stash gold\n\
         design -reset\n\
         read_verilog \"{}\"\n\
         prep -top {}\n\
         rename {} dut\n\
         design -stash gate\n\
         design -reset\n\
         design -copy-from gold spec spec\n\
         design -copy-from gate dut dut\n\
         equiv_make spec dut equiv\n\
         hierarchy -top equiv\n\
         equiv_simple\n\
         equiv_status -assert\n",
        spec.display(),
        spec_top,
        spec_top,
        dut.display(),
        dut_top,
        dut_top
    )
}

fn seq_script(spec: &Path, spec_top: &str, dut: &Path, dut_top: &str, cycles: u32) -> String {
    format!(
        "read_verilog \"{}\"\n\
         prep -top {}\n\
         rename {} spec\n\
         design -stash gold\n\
         design -reset\n\
         read_verilog \"{}\"\n\
         prep -top {}\n\
         rename {} dut\n\
         design -stash gate\n\
         design -reset\n\
         design -copy-from gold spec spec\n\
         design -copy-from gate dut dut\n\
         miter -equiv -flatten spec dut miter\n\
         hierarchy -top miter\n\
         setundef -zero -params\n\
         sat -verify -seq {} -set-init-zero -prove trigger 0\n",
        spec.display(),
        spec_top,
        spec_top,
        dut.display(),
        dut_top,
        dut_top,
        cycles
    )
}

fn write_temp(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, contents).expect("write temp file");
    path
}

fn build_fa8_nir() -> Nir {
    let mut ports = BTreeMap::new();
    ports.insert(
        "a".into(),
        Port {
            dir: PortDirection::Input,
            bits: 8,
            attrs: None,
        },
    );
    ports.insert(
        "b".into(),
        Port {
            dir: PortDirection::Input,
            bits: 8,
            attrs: None,
        },
    );
    ports.insert(
        "cin".into(),
        Port {
            dir: PortDirection::Input,
            bits: 1,
            attrs: None,
        },
    );
    ports.insert(
        "sum".into(),
        Port {
            dir: PortDirection::Output,
            bits: 8,
            attrs: None,
        },
    );
    ports.insert(
        "cout".into(),
        Port {
            dir: PortDirection::Output,
            bits: 1,
            attrs: None,
        },
    );

    let mut nets = BTreeMap::new();
    nets.insert(
        "a".into(),
        Net {
            bits: 8,
            attrs: None,
        },
    );
    nets.insert(
        "b".into(),
        Net {
            bits: 8,
            attrs: None,
        },
    );
    nets.insert(
        "cin".into(),
        Net {
            bits: 1,
            attrs: None,
        },
    );
    nets.insert(
        "sum".into(),
        Net {
            bits: 8,
            attrs: None,
        },
    );
    nets.insert(
        "cout".into(),
        Net {
            bits: 1,
            attrs: None,
        },
    );
    nets.insert(
        "sum_ext".into(),
        Net {
            bits: 9,
            attrs: None,
        },
    );

    let mut nodes = BTreeMap::new();
    nodes.insert(
        "adder".into(),
        node(
            "adder",
            NodeOp::Add,
            9,
            BTreeMap::from([
                ("A".into(), concat(vec![const_bits("0", 1), net("a", 0, 7)])),
                ("B".into(), concat(vec![net("b", 0, 7), net("cin", 0, 0)])),
                ("Y".into(), net("sum_ext", 0, 8)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "sum_slice".into(),
        node(
            "sum_slice",
            NodeOp::Slice,
            8,
            BTreeMap::from([
                ("A".into(), net("sum_ext", 0, 7)),
                ("Y".into(), net("sum", 0, 7)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "cout_slice".into(),
        node(
            "cout_slice",
            NodeOp::Slice,
            1,
            BTreeMap::from([
                ("A".into(), net("sum_ext", 8, 8)),
                ("Y".into(), net("cout", 0, 0)),
            ]),
            None,
        ),
    );

    make_nir("fa8", Module { ports, nets, nodes })
}

fn build_alu32_nir() -> Nir {
    let mut ports = BTreeMap::new();
    ports.insert(
        "a".into(),
        Port {
            dir: PortDirection::Input,
            bits: 32,
            attrs: None,
        },
    );
    ports.insert(
        "b".into(),
        Port {
            dir: PortDirection::Input,
            bits: 32,
            attrs: None,
        },
    );
    ports.insert(
        "op".into(),
        Port {
            dir: PortDirection::Input,
            bits: 2,
            attrs: None,
        },
    );
    ports.insert(
        "y".into(),
        Port {
            dir: PortDirection::Output,
            bits: 32,
            attrs: None,
        },
    );

    let mut nets = BTreeMap::new();
    for (name, bits) in [
        ("a", 32),
        ("b", 32),
        ("op", 2),
        ("y", 32),
        ("sum", 32),
        ("and_ab", 32),
        ("or_ab", 32),
        ("xor_ab", 32),
        ("lo_sel", 32),
        ("hi_sel", 32),
    ] {
        nets.insert(name.into(), Net { bits, attrs: None });
    }

    let mut nodes = BTreeMap::new();
    nodes.insert(
        "add_sum".into(),
        node(
            "add_sum",
            NodeOp::Add,
            32,
            BTreeMap::from([
                ("A".into(), net("a", 0, 31)),
                ("B".into(), net("b", 0, 31)),
                ("Y".into(), net("sum", 0, 31)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "and".into(),
        node(
            "and",
            NodeOp::And,
            32,
            BTreeMap::from([
                ("A".into(), net("a", 0, 31)),
                ("B".into(), net("b", 0, 31)),
                ("Y".into(), net("and_ab", 0, 31)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "or".into(),
        node(
            "or",
            NodeOp::Or,
            32,
            BTreeMap::from([
                ("A".into(), net("a", 0, 31)),
                ("B".into(), net("b", 0, 31)),
                ("Y".into(), net("or_ab", 0, 31)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "xor".into(),
        node(
            "xor",
            NodeOp::Xor,
            32,
            BTreeMap::from([
                ("A".into(), net("a", 0, 31)),
                ("B".into(), net("b", 0, 31)),
                ("Y".into(), net("xor_ab", 0, 31)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "mux_lo".into(),
        node(
            "mux_lo",
            NodeOp::Mux,
            32,
            BTreeMap::from([
                ("A".into(), net("sum", 0, 31)),
                ("B".into(), net("and_ab", 0, 31)),
                ("S".into(), concat(vec![net("op", 0, 0)])),
                ("Y".into(), net("lo_sel", 0, 31)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "mux_hi".into(),
        node(
            "mux_hi",
            NodeOp::Mux,
            32,
            BTreeMap::from([
                ("A".into(), net("or_ab", 0, 31)),
                ("B".into(), net("xor_ab", 0, 31)),
                ("S".into(), concat(vec![net("op", 0, 0)])),
                ("Y".into(), net("hi_sel", 0, 31)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "mux_final".into(),
        node(
            "mux_final",
            NodeOp::Mux,
            32,
            BTreeMap::from([
                ("A".into(), net("lo_sel", 0, 31)),
                ("B".into(), net("hi_sel", 0, 31)),
                ("S".into(), concat(vec![net("op", 1, 1)])),
                ("Y".into(), net("y", 0, 31)),
            ]),
            None,
        ),
    );

    make_nir("alu32", Module { ports, nets, nodes })
}

fn build_counter_nir() -> Nir {
    let mut ports = BTreeMap::new();
    ports.insert(
        "clk".into(),
        Port {
            dir: PortDirection::Input,
            bits: 1,
            attrs: None,
        },
    );
    ports.insert(
        "rst".into(),
        Port {
            dir: PortDirection::Input,
            bits: 1,
            attrs: None,
        },
    );
    ports.insert(
        "count".into(),
        Port {
            dir: PortDirection::Output,
            bits: 4,
            attrs: None,
        },
    );

    let mut nets = BTreeMap::new();
    nets.insert(
        "clk".into(),
        Net {
            bits: 1,
            attrs: None,
        },
    );
    nets.insert(
        "rst".into(),
        Net {
            bits: 1,
            attrs: None,
        },
    );
    nets.insert(
        "count".into(),
        Net {
            bits: 4,
            attrs: None,
        },
    );
    nets.insert(
        "inc".into(),
        Net {
            bits: 4,
            attrs: None,
        },
    );
    nets.insert(
        "next".into(),
        Net {
            bits: 4,
            attrs: None,
        },
    );

    let mut nodes = BTreeMap::new();
    nodes.insert(
        "add".into(),
        node(
            "add",
            NodeOp::Add,
            4,
            BTreeMap::from([
                ("A".into(), net("count", 0, 3)),
                ("B".into(), const_bits("1", 4)),
                ("Y".into(), net("inc", 0, 3)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "mux".into(),
        node(
            "mux",
            NodeOp::Mux,
            4,
            BTreeMap::from([
                ("A".into(), net("inc", 0, 3)),
                ("B".into(), const_bits("0", 4)),
                ("S".into(), net("rst", 0, 0)),
                ("Y".into(), net("next", 0, 3)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "reg".into(),
        node(
            "reg",
            NodeOp::Dff,
            4,
            BTreeMap::from([
                ("D".into(), net("next", 0, 3)),
                ("Q".into(), net("count", 0, 3)),
                ("CLK".into(), net("clk", 0, 0)),
            ]),
            None,
        ),
    );

    make_nir("counter4", Module { ports, nets, nodes })
}

fn build_fifo_nir() -> Nir {
    let mut ports = BTreeMap::new();
    ports.insert(
        "clk".into(),
        Port {
            dir: PortDirection::Input,
            bits: 1,
            attrs: None,
        },
    );
    ports.insert(
        "rst".into(),
        Port {
            dir: PortDirection::Input,
            bits: 1,
            attrs: None,
        },
    );
    ports.insert(
        "enq_valid".into(),
        Port {
            dir: PortDirection::Input,
            bits: 1,
            attrs: None,
        },
    );
    ports.insert(
        "enq_ready".into(),
        Port {
            dir: PortDirection::Output,
            bits: 1,
            attrs: None,
        },
    );
    ports.insert(
        "enq_data".into(),
        Port {
            dir: PortDirection::Input,
            bits: 8,
            attrs: None,
        },
    );
    ports.insert(
        "deq_ready".into(),
        Port {
            dir: PortDirection::Input,
            bits: 1,
            attrs: None,
        },
    );
    ports.insert(
        "deq_valid".into(),
        Port {
            dir: PortDirection::Output,
            bits: 1,
            attrs: None,
        },
    );
    ports.insert(
        "deq_data".into(),
        Port {
            dir: PortDirection::Output,
            bits: 8,
            attrs: None,
        },
    );

    let mut nets = BTreeMap::new();
    for (name, bits) in [
        ("clk", 1),
        ("rst", 1),
        ("enq_valid", 1),
        ("enq_ready", 1),
        ("enq_data", 8),
        ("deq_ready", 1),
        ("deq_valid", 1),
        ("deq_data", 8),
        ("valid_q", 1),
        ("valid_next", 1),
        ("valid_next_pre", 1),
        ("not_valid", 1),
        ("not_deq_ready", 1),
        ("hold_valid", 1),
        ("enq_fire", 1),
        ("data_q", 8),
        ("data_after_enq", 8),
        ("data_next", 8),
    ] {
        nets.insert(name.into(), Net { bits, attrs: None });
    }

    let mut nodes = BTreeMap::new();
    nodes.insert(
        "valid_reg".into(),
        node(
            "valid_reg",
            NodeOp::Dff,
            1,
            BTreeMap::from([
                ("D".into(), net("valid_next", 0, 0)),
                ("Q".into(), net("valid_q", 0, 0)),
                ("CLK".into(), net("clk", 0, 0)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "data_reg".into(),
        node(
            "data_reg",
            NodeOp::Dff,
            8,
            BTreeMap::from([
                ("D".into(), net("data_next", 0, 7)),
                ("Q".into(), net("data_q", 0, 7)),
                ("CLK".into(), net("clk", 0, 0)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "not_valid".into(),
        node(
            "not_valid",
            NodeOp::Not,
            1,
            BTreeMap::from([
                ("A".into(), net("valid_q", 0, 0)),
                ("Y".into(), net("not_valid", 0, 0)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "enq_ready_logic".into(),
        node(
            "enq_ready_logic",
            NodeOp::Or,
            1,
            BTreeMap::from([
                ("A".into(), net("not_valid", 0, 0)),
                ("B".into(), net("deq_ready", 0, 0)),
                ("Y".into(), net("enq_ready", 0, 0)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "not_deq_ready".into(),
        node(
            "not_deq_ready",
            NodeOp::Not,
            1,
            BTreeMap::from([
                ("A".into(), net("deq_ready", 0, 0)),
                ("Y".into(), net("not_deq_ready", 0, 0)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "hold_valid".into(),
        node(
            "hold_valid",
            NodeOp::And,
            1,
            BTreeMap::from([
                ("A".into(), net("valid_q", 0, 0)),
                ("B".into(), net("not_deq_ready", 0, 0)),
                ("Y".into(), net("hold_valid", 0, 0)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "enq_fire".into(),
        node(
            "enq_fire",
            NodeOp::And,
            1,
            BTreeMap::from([
                ("A".into(), net("enq_valid", 0, 0)),
                ("B".into(), net("enq_ready", 0, 0)),
                ("Y".into(), net("enq_fire", 0, 0)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "valid_next_pre".into(),
        node(
            "valid_next_pre",
            NodeOp::Or,
            1,
            BTreeMap::from([
                ("A".into(), net("hold_valid", 0, 0)),
                ("B".into(), net("enq_fire", 0, 0)),
                ("Y".into(), net("valid_next_pre", 0, 0)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "valid_next".into(),
        node(
            "valid_next",
            NodeOp::Mux,
            1,
            BTreeMap::from([
                ("A".into(), net("valid_next_pre", 0, 0)),
                ("B".into(), const_bits("0", 1)),
                ("S".into(), net("rst", 0, 0)),
                ("Y".into(), net("valid_next", 0, 0)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "data_after_enq".into(),
        node(
            "data_after_enq",
            NodeOp::Mux,
            8,
            BTreeMap::from([
                ("A".into(), net("data_q", 0, 7)),
                ("B".into(), net("enq_data", 0, 7)),
                ("S".into(), net("enq_fire", 0, 0)),
                ("Y".into(), net("data_after_enq", 0, 7)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "data_next".into(),
        node(
            "data_next",
            NodeOp::Mux,
            8,
            BTreeMap::from([
                ("A".into(), net("data_after_enq", 0, 7)),
                ("B".into(), const_bits("0", 8)),
                ("S".into(), net("rst", 0, 0)),
                ("Y".into(), net("data_next", 0, 7)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "deq_valid_out".into(),
        node(
            "deq_valid_out",
            NodeOp::Slice,
            1,
            BTreeMap::from([
                ("A".into(), net("valid_q", 0, 0)),
                ("Y".into(), net("deq_valid", 0, 0)),
            ]),
            None,
        ),
    );
    nodes.insert(
        "deq_data_out".into(),
        node(
            "deq_data_out",
            NodeOp::Slice,
            8,
            BTreeMap::from([
                ("A".into(), net("data_q", 0, 7)),
                ("Y".into(), net("deq_data", 0, 7)),
            ]),
            None,
        ),
    );

    make_nir("fifo_small", Module { ports, nets, nodes })
}

fn node(
    uid: &str,
    op: NodeOp,
    width: u32,
    pin_map: BTreeMap<String, BitRef>,
    params: Option<BTreeMap<String, Value>>,
) -> Node {
    Node {
        uid: uid.to_string(),
        op,
        width,
        pin_map,
        params,
        attrs: None,
    }
}

fn net(name: &str, lsb: u32, msb: u32) -> BitRef {
    BitRef::Net(BitRefNet {
        net: name.to_string(),
        lsb,
        msb,
    })
}

fn const_bits(value: &str, width: u32) -> BitRef {
    BitRef::Const(BitRefConst {
        value: value.to_string(),
        width,
    })
}

fn concat(parts: Vec<BitRef>) -> BitRef {
    BitRef::Concat(BitRefConcat { concat: parts })
}

fn make_nir(design_name: &str, module: Module) -> Nir {
    let mut modules = BTreeMap::new();
    modules.insert(design_name.to_string(), module);
    Nir {
        v: "nir-1.1".to_string(),
        design: design_name.to_string(),
        top: design_name.to_string(),
        attrs: None,
        modules,
        generator: None,
        cmdline: None,
        source_digest_sha256: None,
    }
}
