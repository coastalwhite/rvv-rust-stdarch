#! /usr/bin/env python

from enum import Enum, IntEnum
from typing import List, Tuple

class LMUL(Enum):
    MF8 = 1
    MF4 = 2
    MF2 = 4
    M1  = 8
    M2  = 16
    M4  = 32
    M8  = 64

    def to_str(self) -> str:
        return self.name[1:].lower()

class EEW(Enum):
    E8 = 8
    E16 = 16
    E32 = 32
    E64 = 64

ALL_CONFIGURATIONS: List[Tuple[EEW, LMUL]] = []

for eew in EEW:
    for lmul in LMUL:
        if (eew.value << 3) >= lmul.value:
            ALL_CONFIGURATIONS.append((eew, lmul))

class RustTypeVariant(Enum):
    XREG = 0
    VREG = 1
    MASK = 2
    USIZE = 3
    CONST_PTR = 4
    MUT_PTR = 5

class RustType:
    def __init__(self, variant: RustTypeVariant, llvm_type: str, rust_type: str) -> None:
        self.variant = variant
        self.LLVM_TYPE = llvm_type
        self.RUST_TYPE = rust_type

def init_vreg(eew: EEW, lmul: LMUL) -> RustType:
    TY = f"vint{eew.value}m{lmul.to_str()}"
    return RustType(RustTypeVariant.VREG, TY, TY)

def init_mask(lmul: LMUL) -> RustType:
    TY = f"vbool{lmul.to_str()}"
    return RustType(RustTypeVariant.MASK, TY, TY)

def init_const_ptr(bits: EEW) -> RustType:
    return RustType(
        RustTypeVariant.CONST_PTR,
        f"*const i{bits.value}",
        f"*const u{bits.value}"
    )

def init_mut_ptr(bits: EEW) -> RustType:
    return RustType(
        RustTypeVariant.MUT_PTR,
        f"*mut i{bits.value}",
        f"*mut u{bits.value}"
    )

XREG = RustType(RustTypeVariant.XREG, "i64", "u64")
VREG = init_vreg
MASK = init_mask
USIZE = RustType(RustTypeVariant.USIZE, "isize", "usize")
CONST_PTR = init_const_ptr
MUT_PTR = init_mut_ptr

class RustParams:
    def __init__(self, params: List[Tuple[str, RustType]]) -> None:
        self.params = params

    def as_rust_with_types(self):
        return ', '.join(
            [f"{ident}: {ty.RUST_TYPE}" for (ident, ty) in self.params]
        )

    def as_llvm_with_types(self):
        return ', '.join(
            [f"{ident}: {ty.LLVM_TYPE}" for (ident, ty) in self.params]
        )

    def as_llvm_vars(self):
        return ', '.join(
            [
                f"{ident} as {ty.LLVM_TYPE}" if ty.LLVM_TYPE != ty.RUST_TYPE
                else ident
                for (ident, ty) in self.params
            ]
        )

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
        ret_type_as = "" if (
            ret_type == None or 
            ret_type.LLVM_TYPE == ret_type.RUST_TYPE
        ) else f" as {ret_type.RUST_TYPE}",
    )

def parse_type(s: str, elem: EEW, lmul: LMUL) -> RustType:
    s = s.lower().strip()
    
    if s == "xr":
        return XREG
    elif s == "vr":
        return VREG(elem, lmul)
    elif s == "v8":
        return VREG(EEW.E8, lmul)
    elif s == "v16":
        return VREG(EEW.E16, lmul)
    elif s == "v32":
        return VREG(EEW.E32, lmul)
    elif s == "v64":
        return VREG(EEW.E64, lmul)
    elif s == "mask":
        return MASK(lmul)
    elif s == "&e":
        return CONST_PTR(elem)
    elif s == "*e":
        return MUT_PTR(elem)
    elif s == "&b":
        return CONST_PTR(EEW.E8)
    elif s == "*b":
        return MUT_PTR(EEW.E8)
    
    raise Exception(f"Unknown explict type for '{s}'")

def parse_param(s: str, elem: EEW, lmul: LMUL) -> Tuple[Tuple[str, RustType], bool, str]:
    param_end = s.find(',')
    is_end = param_end < 0
    if param_end < 0:
        param_end = s.find(')')
    if param_end < 0:
        raise Exception("Param has no valid end")
    
    param_str = s[:param_end].strip()
    
    explicit_ty_separator = param_str.find(':')
    
    if explicit_ty_separator < 0:
        ident = param_str
        
        if ident.startswith("vs"):
            ty = VREG(elem, lmul)
        elif ident.startswith("rs"):
            ty = XREG
        elif ident == "vl":
            ty = USIZE
        elif ident == "_p":
            ty = VREG(elem, lmul)
        elif ident == "mask":
            ty = MASK(lmul)
        elif ident == "bindex":
            ty = VREG(elem, lmul)
        else:
            raise Exception(f"Unknown implicit type for {ident}")
    else:
        ident = param_str[:explicit_ty_separator]
        ty = parse_type(param_str[explicit_ty_separator+1:], elem, lmul) 
    
    return ((ident, ty), is_end, s[param_end+1:])

def parse_intrinsic_str(s: str, elem: EEW, lmul: LMUL):
    name_end = s.find('(')
    if name_end < 0:
        raise Exception("NO PARAMS")
    name = s[:name_end].strip()
    if len(name) == 0:
        raise Exception("NO NAME")
    name = name.format(
        E = elem.value,
        L = lmul.to_str(),
        B = lmul.value,
    )
    
    s = s[name_end+1:].strip()
    params = []
    
    ((ident, ty), _, s) = parse_param(s, elem, lmul)
    s = s.strip()
    
    if ident == "_p":
        poison = ty
    else:
        poison = None
        params.append((ident, ty))
    
    while True:
        (param, is_end, s) = parse_param(s, elem, lmul)
        s = s.strip()
        
        params.append(param)
        
        if is_end:
            break
            
    sig_end = s.find(':')
    if sig_end < 0:
        raise Exception("No LLVM LINK")
    
    if s.startswith("->"):
        ret_type = parse_type(s[2:sig_end].strip(), elem, lmul)
    else:
        ret_type = None
        
    s = s[sig_end+1:]
    
    llvm_link = s.strip()
    llvm_link = llvm_link.replace("NXV8", f"nxv{lmul.value}i8")
    llvm_link = llvm_link.replace("NXV16", f"nxv{lmul.value}i16")
    llvm_link = llvm_link.replace("NXV32", f"nxv{lmul.value}i32")
    llvm_link = llvm_link.replace("NXV64", f"nxv{lmul.value}i64")
    llvm_link = llvm_link.replace("NXV", f"nxv{lmul.value}i{elem.value}")
    llvm_link = llvm_link.replace("NXM", f"nxv{lmul.value}i1")
    llvm_link = llvm_link.replace("iX", f"i64")

    assert name.lower() == name
    assert llvm_link.lower() == llvm_link
    
    return {
        "name": name,
        "params": RustParams(params),
        "llvm_link": llvm_link,
        "poison": poison,
        "ret_type": ret_type,
    }  
    
def G(intrinsic_strs: List[str]):
    for s in intrinsic_strs:
        for (elem, lmul) in ALL_CONFIGURATIONS:
            intrinsic = parse_intrinsic_str(s, elem, lmul)
            print(intrinsic_fn_str(
                name = intrinsic["name"],
                params = intrinsic["params"],
                llvm_link = intrinsic["llvm_link"],
                poison = intrinsic["poison"],
                ret_type = intrinsic["ret_type"],
            ))

def M(intrinsic_strs: List[str]):
    for s in intrinsic_strs:
        for lmul in LMUL:
            intrinsic = parse_intrinsic_str(s, EEW.E8, lmul)
            print(intrinsic_fn_str(
                name = intrinsic["name"],
                params = intrinsic["params"],
                llvm_link = intrinsic["llvm_link"],
                poison = intrinsic["poison"],
                ret_type = intrinsic["ret_type"],
            ))

print("""
/// AUTOGENERATED FILE
""".strip())

print("""
macro_rules! poison {
    () => {{
        crate::core_arch::simd_llvm::simd_reinterpret(())
    }};
}

#[repr(isize)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SelectedElementWidth {
    E8 = 0b000,
    E16 = 0b001,
    E32 = 0b010,
    E64 = 0b011,
}

#[repr(isize)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VectorLengthMultiplier {
    F8 = 0b101,
    F4 = 0b110,
    F2 = 0b111,
    M1 = 0b000,
    M2 = 0b001,
    M4 = 0b010,
    M8 = 0b011,
}

impl core::marker::ConstParamTy for SelectedElementWidth {}
impl core::marker::ConstParamTy for VectorLengthMultiplier {}

#[inline]
#[target_feature(enable = "v")]
#[rustc_legacy_const_generics(1, 2)]
pub unsafe fn vsetvli<const SEW: SelectedElementWidth, const LMUL: VectorLengthMultiplier>(
    vl: usize,
) -> usize {
    #[allow(improper_ctypes)]
    extern "C" {
        #[link_name = "llvm.riscv.vsetvli.i64"]
        fn _vsetvli(vl: i64, ei: i64, mi: i64) -> i64;
    }

    unsafe { _vsetvli(vl as i64, SEW as isize as i64, LMUL as isize as i64) as usize }
}
""")



# Integer Types
for (elem, lmul) in ALL_CONFIGURATIONS:
    print("""
#[repr(simd, scalable({elem}))]
#[allow(non_camel_case_types)]
pub struct {name} {{
    _ty: [u{elem}],
}}
    """.strip().format(elem = elem.value, name = VREG(elem, lmul).RUST_TYPE))

# Mask Types
for lmul in LMUL:
    print("""
#[repr(simd, scalable(1))]
#[allow(non_camel_case_types)]
pub struct {name} {{
    _ty: [bool],
}}
    """.strip().format(name = MASK(lmul).RUST_TYPE))
        
G([
    "vle{E}_v_i{E}m{L}  (_p,base:&E,     vl)->VR:vle.NXV.iX",
    "vle{E}_v_i{E}m{L}_m(_p,base:&E,mask,vl)->VR:vle.mask.NXV.iX",

    "vse{E}_v_i{E}m{L}  (vs,base:*E,     vl):vse.NXV.iX",
    "vse{E}_v_i{E}m{L}_m(vs,base:*E,mask,vl):vse.mask.NXV.iX",

    "vlse{E}_v_i{E}m{L}  (_p,base:&E,bstride:XR,     vl)->VR:vlse.NXV.iX",
    "vlse{E}_v_i{E}m{L}_m(_p,base:&E,bstride:XR,mask,vl)->VR:vlse.mask.NXV.iX",

    "vsse{E}_v_i{E}m{L}  (vs,base:*E,bstride:XR,     vl):vsse.NXV.iX",
    "vsse{E}_v_i{E}m{L}_m(vs,base:*E,bstride:XR,mask,vl):vsse.mask.NXV.iX",

    "vle{E}ff_v_i{E}m{L}  (_p,base:&E,     vl)->VR:vleff.NXV.iX",
    "vle{E}ff_v_i{E}m{L}_m(_p,base:&E,mask,vl)->VR:vleff.mask.NXV.iX",

    "vloxei8_v_i{E}m{L}   (_p,base:&E,bindex,vl)->V8 :vloxei.NXV.NXV8.iX",
    "vloxei16_v_i{E}m{L}  (_p,base:&E,bindex,vl)->V16:vloxei.NXV.NXV16.iX",
    "vloxei32_v_i{E}m{L}  (_p,base:&E,bindex,vl)->V32:vloxei.NXV.NXV32.iX",
    "vloxei64_v_i{E}m{L}  (_p,base:&E,bindex,vl)->V64:vloxei.NXV.NXV64.iX",

    "vloxei8_v_i{E}m{L}_m (_p,base:&E,bindex,mask,vl)->V8 :vloxei.mask.NXV.NXV8.iX",
    "vloxei16_v_i{E}m{L}_m(_p,base:&E,bindex,mask,vl)->V16:vloxei.mask.NXV.NXV16.iX",
    "vloxei32_v_i{E}m{L}_m(_p,base:&E,bindex,mask,vl)->V32:vloxei.mask.NXV.NXV32.iX",
    "vloxei64_v_i{E}m{L}_m(_p,base:&E,bindex,mask,vl)->V64:vloxei.mask.NXV.NXV64.iX",

    "vluxei8_v_i{E}m{L}   (_p,base:&E,bindex,vl)->V8 :vluxei.NXV.NXV8.iX",
    "vluxei16_v_i{E}m{L}  (_p,base:&E,bindex,vl)->V16:vluxei.NXV.NXV16.iX",
    "vluxei32_v_i{E}m{L}  (_p,base:&E,bindex,vl)->V32:vluxei.NXV.NXV32.iX",
    "vluxei64_v_i{E}m{L}  (_p,base:&E,bindex,vl)->V64:vluxei.NXV.NXV64.iX",

    "vluxei8_v_i{E}m{L}_m (_p,base:&E,bindex,mask,vl)->V8 :vluxei.mask.NXV.NXV8.iX",
    "vluxei16_v_i{E}m{L}_m(_p,base:&E,bindex,mask,vl)->V16:vluxei.mask.NXV.NXV16.iX",
    "vluxei32_v_i{E}m{L}_m(_p,base:&E,bindex,mask,vl)->V32:vluxei.mask.NXV.NXV32.iX",
    "vluxei64_v_i{E}m{L}_m(_p,base:&E,bindex,mask,vl)->V64:vluxei.mask.NXV.NXV64.iX",

    "vadd_vv_i{E}m{L}  (_p,vs2,vs1,     vl)->VR:vadd.NXV.NXV.iX",
    "vadd_vv_i{E}m{L}_m(_p,vs2,vs1,mask,vl)->VR:vadd.mask.NXV.NXV.iX",

    "vsub_vv_i{E}m{L}  (_p,vs2,vs1,     vl)->VR:vsub.NXV.NXV.iX",
    "vsub_vv_i{E}m{L}_m(_p,vs2,vs1,mask,vl)->VR:vsub.mask.NXV.NXV.iX",
])

M([
    "vlm_v_b{B}(base:&B,vl)->mask:vlm.NXM.iX",
    "vsm_v_b{B}(vs:mask,base:*B,vl):vsm.NXM.iX",
])
