use std::collections::BTreeMap;

use anyhow::{anyhow, bail, ensure, Result};

use v2m_formats::nir::{BitRef, BitRefConcat, BitRefConst, BitRefNet, Net as NirNet, NodeOp};

use crate::cells::{expect_param_bool, expect_param_u32, LoweredCell};
use crate::context::Ctx;
use crate::nir_node::NirNode;
use yosys_bridge::loader::Cell;

#[derive(Clone)]
struct CompareSignals {
    eq: BitRef,
    lt: BitRef,
    gt: BitRef,
}

pub fn map_eq(cell: &Cell, ctx: &mut Ctx) -> Result<LoweredCell> {
    let operands = ComparatorOperands::new(cell, ctx)?;
    let mut builder = CellLoweringBuilder::new(ctx.cell_name());

    if operands.width == 0 {
        builder.assign_with_name(ctx.cell_name(), one_bit(), operands.y_bit.clone());
        return Ok(builder.finish());
    }

    let a_bits = builder.materialize_bits(&operands.a_bits);
    let b_bits = builder.materialize_bits(&operands.b_bits);
    let eq_bit = build_equality_bit(&mut builder, &a_bits, &b_bits)?;
    builder.assign_with_name(ctx.cell_name(), eq_bit, operands.y_bit.clone());
    Ok(builder.finish())
}

pub fn map_ne(cell: &Cell, ctx: &mut Ctx) -> Result<LoweredCell> {
    let operands = ComparatorOperands::new(cell, ctx)?;
    let mut builder = CellLoweringBuilder::new(ctx.cell_name());

    if operands.width == 0 {
        builder.assign_with_name(ctx.cell_name(), zero_bit(), operands.y_bit.clone());
        return Ok(builder.finish());
    }

    let a_bits = builder.materialize_bits(&operands.a_bits);
    let b_bits = builder.materialize_bits(&operands.b_bits);
    let eq_bit = build_equality_bit(&mut builder, &a_bits, &b_bits)?;
    builder.emit_named_unop(
        Some(ctx.cell_name()),
        NodeOp::Not,
        eq_bit,
        operands.y_bit.clone(),
    );
    Ok(builder.finish())
}

pub fn map_lt(cell: &Cell, ctx: &mut Ctx) -> Result<LoweredCell> {
    let operands = ComparatorOperands::new(cell, ctx)?;
    let mut builder = CellLoweringBuilder::new(ctx.cell_name());

    if operands.width == 0 {
        builder.assign_with_name(ctx.cell_name(), zero_bit(), operands.y_bit.clone());
        return Ok(builder.finish());
    }

    let a_bits = builder.materialize_bits(&operands.a_bits);
    let b_bits = builder.materialize_bits(&operands.b_bits);
    let cmp = build_unsigned_compare(&mut builder, &a_bits, &b_bits)?;
    let result = if operands.is_signed_relational() {
        let terms = build_signed_terms(
            &mut builder,
            &a_bits,
            &b_bits,
            operands.a_signed,
            operands.b_signed,
        );
        signed_lt(&mut builder, &cmp, &terms)
    } else {
        cmp.lt
    };
    builder.assign_with_name(ctx.cell_name(), result, operands.y_bit.clone());
    Ok(builder.finish())
}

pub fn map_le(cell: &Cell, ctx: &mut Ctx) -> Result<LoweredCell> {
    let operands = ComparatorOperands::new(cell, ctx)?;
    let mut builder = CellLoweringBuilder::new(ctx.cell_name());

    if operands.width == 0 {
        builder.assign_with_name(ctx.cell_name(), one_bit(), operands.y_bit.clone());
        return Ok(builder.finish());
    }

    let a_bits = builder.materialize_bits(&operands.a_bits);
    let b_bits = builder.materialize_bits(&operands.b_bits);
    let cmp = build_unsigned_compare(&mut builder, &a_bits, &b_bits)?;
    let (lt_bit, eq_bit) = if operands.is_signed_relational() {
        let terms = build_signed_terms(
            &mut builder,
            &a_bits,
            &b_bits,
            operands.a_signed,
            operands.b_signed,
        );
        let lt = signed_lt(&mut builder, &cmp, &terms);
        let eq = builder.emit_binop(NodeOp::And, cmp.eq.clone(), terms.sign_same.clone(), 1);
        (lt, eq)
    } else {
        (cmp.lt.clone(), cmp.eq.clone())
    };
    let le_bit = builder.emit_binop(NodeOp::Or, lt_bit, eq_bit, 1);
    builder.assign_with_name(ctx.cell_name(), le_bit, operands.y_bit.clone());
    Ok(builder.finish())
}

pub fn map_gt(cell: &Cell, ctx: &mut Ctx) -> Result<LoweredCell> {
    let operands = ComparatorOperands::new(cell, ctx)?;
    let mut builder = CellLoweringBuilder::new(ctx.cell_name());

    if operands.width == 0 {
        builder.assign_with_name(ctx.cell_name(), zero_bit(), operands.y_bit.clone());
        return Ok(builder.finish());
    }

    let a_bits = builder.materialize_bits(&operands.a_bits);
    let b_bits = builder.materialize_bits(&operands.b_bits);
    let cmp = build_unsigned_compare(&mut builder, &a_bits, &b_bits)?;
    let result = if operands.is_signed_relational() {
        let terms = build_signed_terms(
            &mut builder,
            &a_bits,
            &b_bits,
            operands.a_signed,
            operands.b_signed,
        );
        signed_gt(&mut builder, &cmp, &terms)
    } else {
        cmp.gt
    };
    builder.assign_with_name(ctx.cell_name(), result, operands.y_bit.clone());
    Ok(builder.finish())
}

pub fn map_ge(cell: &Cell, ctx: &mut Ctx) -> Result<LoweredCell> {
    let operands = ComparatorOperands::new(cell, ctx)?;
    let mut builder = CellLoweringBuilder::new(ctx.cell_name());

    if operands.width == 0 {
        builder.assign_with_name(ctx.cell_name(), one_bit(), operands.y_bit.clone());
        return Ok(builder.finish());
    }

    let a_bits = builder.materialize_bits(&operands.a_bits);
    let b_bits = builder.materialize_bits(&operands.b_bits);
    let cmp = build_unsigned_compare(&mut builder, &a_bits, &b_bits)?;
    let (gt_bit, eq_bit) = if operands.is_signed_relational() {
        let terms = build_signed_terms(
            &mut builder,
            &a_bits,
            &b_bits,
            operands.a_signed,
            operands.b_signed,
        );
        let gt = signed_gt(&mut builder, &cmp, &terms);
        let eq = builder.emit_binop(NodeOp::And, cmp.eq.clone(), terms.sign_same.clone(), 1);
        (gt, eq)
    } else {
        (cmp.gt.clone(), cmp.eq.clone())
    };
    let ge_bit = builder.emit_binop(NodeOp::Or, gt_bit, eq_bit, 1);
    builder.assign_with_name(ctx.cell_name(), ge_bit, operands.y_bit.clone());
    Ok(builder.finish())
}

struct ComparatorOperands {
    a_bits: Vec<BitRef>,
    b_bits: Vec<BitRef>,
    y_bit: BitRef,
    width: usize,
    a_signed: bool,
    b_signed: bool,
}

impl ComparatorOperands {
    fn new(cell: &Cell, ctx: &mut Ctx) -> Result<Self> {
        let a = ctx.bitref(cell, "A")?;
        let b = ctx.bitref(cell, "B")?;
        let y = ctx.bitref(cell, "Y")?;

        let a_width = expect_param_u32(cell, ctx, "A_WIDTH")?;
        let b_width = expect_param_u32(cell, ctx, "B_WIDTH")?;
        let y_width = expect_param_u32(cell, ctx, "Y_WIDTH")?;

        ctx.expect_width("A", &a, a_width)?;
        ctx.expect_width("B", &b, b_width)?;
        ctx.expect_width("Y", &y, y_width)?;

        if y_width != 1 {
            bail!(
                "cell `{}` in module `{}` must have Y_WIDTH=1 but found {y_width}",
                ctx.cell_name(),
                ctx.module_name()
            );
        }

        let a_signed = expect_param_bool(cell, ctx, "A_SIGNED")?;
        let b_signed = expect_param_bool(cell, ctx, "B_SIGNED")?;

        let mut a_bits = bitref_to_bits(&a)?;
        ensure!(
            a_bits.len() == a_width as usize,
            "pin `A` on cell `{}` in module `{}` has width {} but expected {}",
            ctx.cell_name(),
            ctx.module_name(),
            a_bits.len(),
            a_width
        );

        let mut b_bits = bitref_to_bits(&b)?;
        ensure!(
            b_bits.len() == b_width as usize,
            "pin `B` on cell `{}` in module `{}` has width {} but expected {}",
            ctx.cell_name(),
            ctx.module_name(),
            b_bits.len(),
            b_width
        );

        let y_bits = bitref_to_bits(&y)?;
        ensure!(
            y_bits.len() == 1,
            "pin `Y` on cell `{}` in module `{}` must have a single bit output",
            ctx.cell_name(),
            ctx.module_name()
        );

        let width = std::cmp::max(a_bits.len(), b_bits.len());
        a_bits = extend_bits(a_bits, width, a_signed);
        b_bits = extend_bits(b_bits, width, b_signed);

        Ok(Self {
            a_bits,
            b_bits,
            y_bit: y_bits[0].clone(),
            width,
            a_signed,
            b_signed,
        })
    }

    fn is_signed_relational(&self) -> bool {
        self.a_signed || self.b_signed
    }
}

struct SignedTerms {
    a_sign: BitRef,
    b_sign: BitRef,
    sign_same: BitRef,
    not_a_sign: BitRef,
    not_b_sign: BitRef,
}

fn build_signed_terms(
    builder: &mut CellLoweringBuilder,
    a_bits: &[BitRef],
    b_bits: &[BitRef],
    a_signed: bool,
    b_signed: bool,
) -> SignedTerms {
    let a_sign = if a_signed {
        a_bits.last().cloned().unwrap_or_else(zero_bit)
    } else {
        builder
            .materialize_bits(&[zero_bit()])
            .into_iter()
            .next()
            .unwrap()
    };
    let b_sign = if b_signed {
        b_bits.last().cloned().unwrap_or_else(zero_bit)
    } else {
        builder
            .materialize_bits(&[zero_bit()])
            .into_iter()
            .next()
            .unwrap()
    };
    let sign_diff = builder.emit_binop(NodeOp::Xor, a_sign.clone(), b_sign.clone(), 1);
    let sign_same = invert_bit(builder, &sign_diff);
    let not_a_sign = invert_bit(builder, &a_sign);
    let not_b_sign = invert_bit(builder, &b_sign);
    SignedTerms {
        a_sign,
        b_sign,
        sign_same,
        not_a_sign,
        not_b_sign,
    }
}

fn signed_lt(
    builder: &mut CellLoweringBuilder,
    cmp: &CompareSignals,
    terms: &SignedTerms,
) -> BitRef {
    let diff_term = builder.emit_binop(
        NodeOp::And,
        terms.a_sign.clone(),
        terms.not_b_sign.clone(),
        1,
    );
    let mag_term = builder.emit_binop(NodeOp::And, terms.sign_same.clone(), cmp.lt.clone(), 1);
    builder.emit_binop(NodeOp::Or, diff_term, mag_term, 1)
}

fn invert_bit(builder: &mut CellLoweringBuilder, bit: &BitRef) -> BitRef {
    match bit {
        BitRef::Const(constant) => match constant.value.as_str() {
            "0b0" => one_bit(),
            "0b1" => zero_bit(),
            _ => builder.emit_unop(NodeOp::Not, bit.clone(), 1),
        },
        _ => builder.emit_unop(NodeOp::Not, bit.clone(), 1),
    }
}

fn signed_gt(
    builder: &mut CellLoweringBuilder,
    cmp: &CompareSignals,
    terms: &SignedTerms,
) -> BitRef {
    let diff_term = builder.emit_binop(
        NodeOp::And,
        terms.not_a_sign.clone(),
        terms.b_sign.clone(),
        1,
    );
    let mag_term = builder.emit_binop(NodeOp::And, terms.sign_same.clone(), cmp.gt.clone(), 1);
    builder.emit_binop(NodeOp::Or, diff_term, mag_term, 1)
}

struct CellLoweringBuilder<'a> {
    cell_name: &'a str,
    tmp_net_index: usize,
    tmp_node_index: usize,
    lowered: LoweredCell,
}

impl<'a> CellLoweringBuilder<'a> {
    fn new(cell_name: &'a str) -> Self {
        Self {
            cell_name,
            tmp_net_index: 0,
            tmp_node_index: 0,
            lowered: LoweredCell::default(),
        }
    }

    fn finish(self) -> LoweredCell {
        self.lowered
    }

    fn alloc_tmp_bitref(&mut self, width: u32) -> BitRef {
        assert!(width > 0, "temporary nets must have positive width");
        let name = format!("{}__v2m_cmp_net{}", self.cell_name, self.tmp_net_index);
        self.tmp_net_index += 1;
        self.lowered.push_net(
            name.clone(),
            NirNet {
                bits: width,
                attrs: None,
            },
        );
        BitRef::Net(BitRefNet {
            net: name,
            lsb: 0,
            msb: width - 1,
        })
    }

    fn emit_node(&mut self, name: Option<String>, node: NirNode) -> String {
        let node_name = match name {
            Some(name) => name,
            None => {
                let generated = format!("{}__v2m_cmp_node{}", self.cell_name, self.tmp_node_index);
                self.tmp_node_index += 1;
                generated
            }
        };
        self.lowered.push_node(node_name.clone(), node);
        node_name
    }

    fn emit_binop(&mut self, op: NodeOp, left: BitRef, right: BitRef, width: u32) -> BitRef {
        let out = self.alloc_tmp_bitref(width);
        self.emit_named_binop(None, op, left, right, out.clone());
        out
    }

    fn emit_named_binop(
        &mut self,
        name: Option<&str>,
        op: NodeOp,
        left: BitRef,
        right: BitRef,
        out: BitRef,
    ) {
        let mut pins = BTreeMap::new();
        pins.insert("A".to_string(), left);
        pins.insert("B".to_string(), right);
        pins.insert("Y".to_string(), out.clone());
        let node = NirNode::new(op, Ctx::bitref_width(&out), pins);
        self.emit_node(name.map(|n| n.to_string()), node);
    }

    fn emit_unop(&mut self, op: NodeOp, input: BitRef, width: u32) -> BitRef {
        let out = self.alloc_tmp_bitref(width);
        self.emit_named_unop(None, op, input, out.clone());
        out
    }

    fn emit_named_unop(
        &mut self,
        name: Option<&str>,
        op: NodeOp,
        input: BitRef,
        out: BitRef,
    ) -> BitRef {
        let mut pins = BTreeMap::new();
        pins.insert("A".to_string(), input);
        pins.insert("Y".to_string(), out.clone());
        let node = NirNode::new(op, Ctx::bitref_width(&out), pins);
        self.emit_node(name.map(|n| n.to_string()), node);
        out
    }

    fn emit_slice(&mut self, name: Option<&str>, input: BitRef, out: BitRef) {
        let mut pins = BTreeMap::new();
        pins.insert("A".to_string(), input);
        pins.insert("Y".to_string(), out.clone());
        let node = NirNode::new(NodeOp::Slice, Ctx::bitref_width(&out), pins);
        self.emit_node(name.map(|n| n.to_string()), node);
    }

    fn materialize_bits(&mut self, bits: &[BitRef]) -> Vec<BitRef> {
        let mut result = Vec::with_capacity(bits.len());
        for bit in bits {
            let target = self.alloc_tmp_bitref(1);
            self.emit_slice(None, bit.clone(), target.clone());
            result.push(target);
        }
        result
    }

    fn reduce_bits(&mut self, bits: &[BitRef], op: NodeOp) -> Result<BitRef> {
        match bits.len() {
            0 => match op {
                NodeOp::And => Ok(one_bit()),
                NodeOp::Or => Ok(zero_bit()),
                _ => bail!("unsupported reduction op"),
            },
            1 => Ok(bits[0].clone()),
            _ => {
                let mid = bits.len() / 2;
                let left = self.reduce_bits(&bits[..mid], op.clone())?;
                let right = self.reduce_bits(&bits[mid..], op.clone())?;
                Ok(self.emit_binop(op, left, right, 1))
            }
        }
    }

    fn assign_with_name(&mut self, name: &str, source: BitRef, target: BitRef) {
        self.emit_slice(Some(name), source, target);
    }
}

fn build_unsigned_compare(
    builder: &mut CellLoweringBuilder,
    a_bits: &[BitRef],
    b_bits: &[BitRef],
) -> Result<CompareSignals> {
    ensure!(
        a_bits.len() == b_bits.len(),
        "operand width mismatch in comparator lowering"
    );
    match a_bits.len() {
        0 => Ok(CompareSignals {
            eq: one_bit(),
            lt: zero_bit(),
            gt: zero_bit(),
        }),
        1 => {
            let a_bit = a_bits[0].clone();
            let b_bit = b_bits[0].clone();
            let eq = builder.emit_binop(NodeOp::Xnor, a_bit.clone(), b_bit.clone(), 1);
            let not_a = builder.emit_unop(NodeOp::Not, a_bit.clone(), 1);
            let not_b = builder.emit_unop(NodeOp::Not, b_bit.clone(), 1);
            let lt = builder.emit_binop(NodeOp::And, not_a, b_bit, 1);
            let gt = builder.emit_binop(NodeOp::And, a_bit, not_b, 1);
            Ok(CompareSignals { eq, lt, gt })
        }
        _ => {
            let mid = a_bits.len() / 2;
            let low = build_unsigned_compare(builder, &a_bits[..mid], &b_bits[..mid])?;
            let high = build_unsigned_compare(builder, &a_bits[mid..], &b_bits[mid..])?;

            let CompareSignals {
                eq: eq_low,
                lt: lt_low,
                gt: gt_low,
            } = low;
            let CompareSignals {
                eq: eq_high,
                lt: lt_high,
                gt: gt_high,
            } = high;

            let eq = builder.emit_binop(NodeOp::And, eq_high.clone(), eq_low.clone(), 1);
            let lt_high_term = builder.emit_binop(NodeOp::And, eq_high.clone(), lt_low.clone(), 1);
            let lt = builder.emit_binop(NodeOp::Or, lt_high.clone(), lt_high_term, 1);
            let gt_high_term = builder.emit_binop(NodeOp::And, eq_high.clone(), gt_low.clone(), 1);
            let gt = builder.emit_binop(NodeOp::Or, gt_high.clone(), gt_high_term, 1);
            Ok(CompareSignals { eq, lt, gt })
        }
    }
}

fn build_equality_bit(
    builder: &mut CellLoweringBuilder,
    a_bits: &[BitRef],
    b_bits: &[BitRef],
) -> Result<BitRef> {
    ensure!(
        a_bits.len() == b_bits.len(),
        "operand width mismatch in equality lowering",
    );
    if a_bits.is_empty() {
        return Ok(one_bit());
    }

    let mut xnor_bits = Vec::with_capacity(a_bits.len());
    for (a_bit, b_bit) in a_bits.iter().zip(b_bits.iter()) {
        xnor_bits.push(builder.emit_binop(NodeOp::Xnor, a_bit.clone(), b_bit.clone(), 1));
    }
    builder.reduce_bits(&xnor_bits, NodeOp::And)
}

fn bitref_to_bits(bitref: &BitRef) -> Result<Vec<BitRef>> {
    match bitref {
        BitRef::Net(BitRefNet { net, lsb, msb }) => {
            let mut bits = Vec::with_capacity((*msb - *lsb + 1) as usize);
            for bit in *lsb..=*msb {
                bits.push(BitRef::Net(BitRefNet {
                    net: net.clone(),
                    lsb: bit,
                    msb: bit,
                }));
            }
            Ok(bits)
        }
        BitRef::Const(constant) => const_to_bits(constant),
        BitRef::Concat(BitRefConcat { concat }) => {
            let mut result = Vec::new();
            for part in concat.iter().rev() {
                result.extend(bitref_to_bits(part)?);
            }
            Ok(result)
        }
    }
}

fn const_to_bits(constant: &BitRefConst) -> Result<Vec<BitRef>> {
    let text = constant
        .value
        .strip_prefix("0b")
        .ok_or_else(|| anyhow!("constant value `{}` must be binary", constant.value))?;
    let mut bits = Vec::with_capacity(constant.width as usize);
    for ch in text.chars().rev() {
        match ch {
            '0' => bits.push(zero_bit()),
            '1' => bits.push(one_bit()),
            other => bail!("unsupported constant bit `{other}` in comparator lowering"),
        }
    }
    ensure!(
        bits.len() == constant.width as usize,
        "constant bit count does not match declared width"
    );
    Ok(bits)
}

fn extend_bits(mut bits: Vec<BitRef>, target: usize, signed: bool) -> Vec<BitRef> {
    let current = bits.len();
    if current >= target {
        bits
    } else {
        let fill_count = target - current;
        if signed && current > 0 {
            let sign_bit = bits[current - 1].clone();
            bits.extend(std::iter::repeat_with(|| sign_bit.clone()).take(fill_count));
        } else {
            bits.extend(std::iter::repeat_with(zero_bit).take(fill_count));
        }
        bits
    }
}

fn zero_bit() -> BitRef {
    BitRef::Const(BitRefConst {
        value: "0b0".to_string(),
        width: 1,
    })
}

fn one_bit() -> BitRef {
    BitRef::Const(BitRefConst {
        value: "0b1".to_string(),
        width: 1,
    })
}
