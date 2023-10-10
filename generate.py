#! /usr/bin/env python

from enum import Enum
from typing import List, Tuple

LMULS = [1, 2, 4, 8, 16, 32]
EEWS = [8, 16, 32, 64]

class RustTypeVariant(Enum):
    XREG = 0
    VREG = 1
    CONST_PTR = 2
    MUT_PTR = 3

class RustType:
    def __init__(self, variant: RustTypeVariant, llvm_type: str, rust_type: str) -> None:
        self.variant = variant
        self.LLVM_TYPE = llvm_type
        self.RUST_TYPE = rust_type

def lmul_to_str(lmul: int) -> str:
    return {
        1: "f8",
        2: "f4",
        3: "f2",
        4: "1",
        8: "2",
        16: "4",
        32: "8",
    }[lmul]


def init_vreg(eew: int, lmul: int) -> RustType:
    assert eew in EEWS
    assert lmul in LMULS

    LMUL_STR = lmul_to_str(lmul)
    TY = f"vint{eew}m{LMUL_STR}"
    return RustType(RustTypeVariant.VREG, TY, TY)

def init_const_ptr(bits: int) -> RustType:
    assert bits in EEWS
    return RustType(RustTypeVariant.CONST_PTR, f"*const i{bits}", f"*const u{bits}")

def init_mut_ptr(bits: int) -> RustType:
    assert bits in EEWS
    return RustType(RustTypeVariant.MUT_PTR, f"*mut i{bits}", f"*mut u{bits}")

XREG = RustType(RustTypeVariant.XREG, "i64", "u64")
VREG = init_vreg
CONST_PTR = init_const_ptr
MUT_PTR = init_mut_ptr

class RustParams:
    def __init__(self, params: List[Tuple[str, RustType]]) -> None:
        self.params = params

    def has_numbered_xregs(self) -> bool:
        return [param[1].variant == RustTypeVariant.XREG for param in self.params].count(True) > 1

    def has_numbered_vregs(self) -> bool:
        return [param[1].variant == RustTypeVariant.VREG for param in self.params].count(True) > 1

    def get_var_ident(self, idx: int) -> str:
        xreg_idx = 0
        vreg_idx = 0

        has_numbered_xregs = self.has_numbered_xregs()
        has_numbered_vregs = self.has_numbered_vregs()

        for i, (ident, ty) in enumerate(self.params):
            assert not(ty.variant != RustTypeVariant.XREG and ident == "vl")

            if i == idx:
                if ident == "vl":
                    return "vl"

                num = ""
                if ty.variant == RustTypeVariant.VREG and has_numbered_vregs:
                    num = f"{vreg_idx}"
                if ty.variant == RustTypeVariant.XREG and has_numbered_xregs:
                    num = f"{xreg_idx}"

                return f"{ident}{num}" 

            if ident == "vl":
                continue

            if ty.variant == RustTypeVariant.VREG and has_numbered_vregs:
                vreg_idx += 1
            if ty.variant == RustTypeVariant.XREG and has_numbered_xregs:
                xreg_idx += 1

        raise Exception("Variable index out of range")

    def as_rust_with_types(self):
        s = []

        for i, (_, ty) in enumerate(self.params):
            s.append(f"{self.get_var_ident(i)}: {ty.RUST_TYPE}")

        return ', '.join(s)

    def as_llvm_with_types(self):
        s = []

        for i, (_, ty) in enumerate(self.params):
            s.append(f"{self.get_var_ident(i)}: {ty.LLVM_TYPE}")

        return ', '.join(s)

    def as_llvm_vars(self):
        s = []

        for i, (_, ty) in enumerate(self.params):
            s.append(f"{self.get_var_ident(i)} as {ty.LLVM_TYPE}")

        return ', '.join(s)

    def is_empty(self) -> bool:
        return len(self.params) == 0

def intrinsic_fn_str(
    name: str,
    params: RustParams,
    llvm_link: str,
    poison: RustType | None = None,
    ret_type: RustType | None = None
) -> str:
    assert not(poison != None and params.is_empty())

    INTRINSIC_FN="""
#[inline]
#[target_feature(enable = "v")]
pub unsafe fn {name}({params}) {ret_type}{{
    #[allow(improper_ctypes)]
    extern "unadjusted" {{
        #[link_name = "llvm.riscv.{llvm_link}"]
        fn _{name}({poison}{llvm_params}){llvm_ret_type};
    }}
    
    unsafe {{
        _{name}({poison_param}{param_vars}){ret_type_as}
    }}
}}
    """.strip()

    return INTRINSIC_FN.format(
        name = name,
        params = params.as_rust_with_types(),
        ret_type = "" if ret_type == None else f"-> {ret_type.RUST_TYPE} ",
        llvm_link = llvm_link,
        poison = "" if not(poison) else f"_poison: {poison.LLVM_TYPE}, ",
        llvm_params = params.as_llvm_with_types(),
        llvm_ret_type = "" if ret_type == None else f" -> {ret_type.LLVM_TYPE}",
        poison_param = "" if not(poison) else f"poison!(), ",
        param_vars = params.as_llvm_vars(),
        ret_type_as = "" if ret_type == None else f" as {ret_type.RUST_TYPE}",
    )

def struct_str(
    elem: int,
    lmul: int,
) -> str:
    STRUCT="""
#[repr(simd, scalable({elem})]
#[allow(non_camel_case_types)]
pub struct {name} {{
    _ty: [u{elem}],
}}
    """.strip()

    return STRUCT.format(
        elem = elem,
        name = VREG(elem, lmul).RUST_TYPE,
    )

def generate_unit_strided_memory():
    for elem in EEWS:
        for lmul in LMULS:
            print(intrinsic_fn_str(
                name = f"vle{elem}_m{lmul}_v",
                params = RustParams([("base", CONST_PTR(elem)), ("vl", XREG)]),
                llvm_link = f"vle.nxv{lmul}i{elem}.i64",
                poison = VREG(elem, lmul),
                ret_type = VREG(elem, lmul),
            ))

            print(intrinsic_fn_str(
                name = f"vse{elem}_m{lmul}_v",
                params = RustParams([("rd", VREG(elem, lmul)), ("base", CONST_PTR(elem)), ("vl", XREG)]),
                llvm_link = f"vse.nxv{lmul}i{elem}.i64",
                poison = None,
                ret_type = None,
            ))

            print("")


print("""
macro_rules! undef {
    () => {{
        simd_reinterpret(())
    }};
}
""")

for elem in EEWS:
    for lmul in LMULS:
        print(struct_str(elem, lmul))

generate_unit_strided_memory()