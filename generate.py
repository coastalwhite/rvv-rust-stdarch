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
        4:  "1",
        8:  "2",
        16: "4",
        32: "8",
    }[lmul]

def init_vreg(eew: int, lmul: int) -> RustType:
    assert eew in EEWS
    assert lmul in LMULS

    TY = f"vint{eew}m{lmul_to_str(lmul)}"
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

def parse_type(s: str, elem: int, lmul) -> RustType:
    s = s.lower().strip()
    
    if s == "xr":
        return XREG
    elif s == "vr":
        return VREG(elem, lmul)
    elif s == "&e":
        return CONST_PTR(elem)
    elif s == "*e":
        return MUT_PTR(elem)
    
    raise Exception(f"Unknown explict type for '{s}'")

def parse_param(s: str, elem: int, lmul: int) -> Tuple[Tuple[str, RustType], bool, str]:
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
            ty = XREG
        elif ident == "_p":
            ty = VREG(elem, lmul)
        else:
            raise Exception(f"Unknown implicit type for {ident}")
    else:
        ident = param_str[:explicit_ty_separator]
        ty = parse_type(param_str[explicit_ty_separator+1:], elem, lmul) 
    
    return ((ident, ty), is_end, s[param_end+1:])

def parse_intrinsic_str(s: str, elem: int, lmul: int):
    name_end = s.find('(')
    if name_end < 0:
        raise Exception("NO PARAMS")
    name = s[:name_end].strip()
    if len(name) == 0:
        raise Exception("NO NAME")
    name = name.format(E = elem, L = lmul)
    
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
    llvm_link = llvm_link.replace("NXV", f"nxv{lmul}i{elem}")
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
        for elem in EEWS:
            for lmul in LMULS:
                intrinsic = parse_intrinsic_str(s, elem, lmul)
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
macro_rules! undef {
    () => {{
        simd_reinterpret(())
    }};
}
""")

for elem in EEWS:
    for lmul in LMULS:
        print("""
#[repr(simd, scalable({elem})]
#[allow(non_camel_case_types)]
pub struct {name} {{
    _ty: [u{elem}],
}}
        """.strip().format(elem = elem, name = VREG(elem, lmul).RUST_TYPE))
        
G([
    "vle{E}_m{L}_v(_p,base:&E,vl)->VR:vle.NXV.iX",
    "vse{E}_m{L}_v(vs,base:*E,vl):vse.NXV.iX",
    "vlse{E}_m{L}_v(_p,base:&E,bstride:XR,vl)->VR:vlse.NXV.iX",
    "vsse{E}_m{L}_v(vs,base:*E,bstride:XR,vl):vsse.NXV.iX",
    "vle{E}ff_m{L}_v(_p,base:&E,vl)->VR:vleff.NXV.iX",
    "vadd_e{E}_m{L}_vv(_p,vs2,vs1,vl)->VR:vadd.NXV.NXV.iX",
    "vsub_e{E}_m{L}_vv(_p,vs2,vs1,vl)->VR:vsub.NXV.NXV.iX",
])
