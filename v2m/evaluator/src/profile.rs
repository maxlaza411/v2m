use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelKind {
    Const,
    Slice,
    Cat,
    Not,
    And,
    Or,
    Xor,
    Xnor,
    Mux,
    Add,
    Sub,
}

impl fmt::Display for KernelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            KernelKind::Const => "const",
            KernelKind::Slice => "slice",
            KernelKind::Cat => "cat",
            KernelKind::Not => "not",
            KernelKind::And => "and",
            KernelKind::Or => "or",
            KernelKind::Xor => "xor",
            KernelKind::Xnor => "xnor",
            KernelKind::Mux => "mux",
            KernelKind::Add => "add",
            KernelKind::Sub => "sub",
        };
        f.write_str(text)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct KernelReport {
    pub kind: KernelKind,
    pub ops: u64,
    pub total_time_ns: u128,
    pub average_time_ns: u128,
    pub bytes_moved: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProfileReport {
    pub total_ops: u64,
    pub total_bytes: u64,
    pub total_time_ns: u128,
    pub kernels: Vec<KernelReport>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct Profiler {
    kernels: BTreeMap<KernelKind, KernelMetrics>,
    total_ops: u64,
    total_bytes: u64,
    total_time: Duration,
}

#[derive(Debug, Default, Clone)]
struct KernelMetrics {
    ops: u64,
    total_time: Duration,
    bytes: u64,
}

impl Profiler {
    pub(crate) fn record(&mut self, kind: KernelKind, bytes: u64, duration: Duration) {
        let entry = self.kernels.entry(kind).or_default();
        entry.ops = entry.ops.saturating_add(1);
        entry.bytes = entry.bytes.saturating_add(bytes);
        entry.total_time += duration;

        self.total_ops = self.total_ops.saturating_add(1);
        self.total_bytes = self.total_bytes.saturating_add(bytes);
        self.total_time += duration;
    }

    pub(crate) fn report(&self) -> ProfileReport {
        let mut kernels: Vec<KernelReport> = self
            .kernels
            .iter()
            .filter_map(|(kind, metrics)| {
                if metrics.ops == 0 {
                    return None;
                }
                let total_time_ns = metrics.total_time.as_nanos();
                let average_time_ns = if metrics.ops == 0 {
                    0
                } else {
                    total_time_ns / metrics.ops as u128
                };
                Some(KernelReport {
                    kind: *kind,
                    ops: metrics.ops,
                    total_time_ns,
                    average_time_ns,
                    bytes_moved: metrics.bytes,
                })
            })
            .collect();

        kernels.sort_by(|a, b| b.total_time_ns.cmp(&a.total_time_ns));

        ProfileReport {
            total_ops: self.total_ops,
            total_bytes: self.total_bytes,
            total_time_ns: self.total_time.as_nanos(),
            kernels,
        }
    }
}

impl From<&Profiler> for ProfileReport {
    fn from(profiler: &Profiler) -> Self {
        profiler.report()
    }
}
