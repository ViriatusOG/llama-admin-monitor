use anyhow::Result;
use std::collections::BTreeMap;

use super::{GpuBackend, GpuMetrics};

pub struct DummyBackend;

impl GpuBackend for DummyBackend {
    fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>> {
        Ok(BTreeMap::new())
    }

    fn name(&self) -> &str {
        "none"
    }
}
