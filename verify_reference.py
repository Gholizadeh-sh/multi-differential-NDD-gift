"""Compare against the designers' unmodified C++ source, all 28 round prefixes.

Requires g++; runs offline using the vendored, pinned source.
"""
from pathlib import Path
import json
import random
import shutil
import subprocess
import tempfile
from gift64 import encrypt

ROOT = Path(__file__).resolve().parent


def main():
    compiler = shutil.which("g++")
    if not compiler:
        raise SystemExit("g++ missing; install build-essential (Debian/Ubuntu)")
    rng = random.Random(123456)
    cases = [(rng.getrandbits(64),rng.getrandbits(128),r) for r in range(1,29) for _ in range(20)]
    source = ROOT/"third_party/giftcipher/GIFT64-128_cipher.cpp"
    harness = r'''
#define main official_reference_main
#include "GIFT64-128_cipher.cpp"
#undef main
#include <sstream>
int main() {
    unsigned long long pt; std::string keyhex; int rounds;
    while (std::cin >> std::hex >> pt >> keyhex >> std::dec >> rounds) {
        unsigned char p[16], k[32];
        for(int i=0;i<16;i++) p[i]=(pt>>(4*i))&15;
        for(int i=0;i<32;i++) {
            char c=keyhex[31-i]; k[i]=(c<='9')?c-'0':c-'a'+10;
        }
        std::ostringstream sink; auto old=std::cout.rdbuf(sink.rdbuf());
        enc64(p,k,rounds,false); std::cout.rdbuf(old);
        unsigned long long ct=0;
        for(int i=0;i<16;i++) ct|=((unsigned long long)p[i])<<(4*i);
        std::cout<<std::hex<<ct<<std::endl;
    }
}
'''
    with tempfile.TemporaryDirectory() as d:
        d=Path(d)
        (d/"harness.cpp").write_text(harness)
        subprocess.run([compiler,"-O2","-std=c++11","-I",str(source.parent),str(d/"harness.cpp"),"-o",str(d/"oracle")],check=True)
        result=subprocess.run([str(d/"oracle")],input="".join(f"{p:016x} {k:032x} {r}\n" for p,k,r in cases),text=True,capture_output=True,check=True)
    actual=[int(s,16) for s in result.stdout.splitlines()]
    assert len(actual)==len(cases)
    for i,((p,k,r),ct) in enumerate(zip(cases,actual)):
        if encrypt(p,k,r)!=ct:
            raise AssertionError(f"Official C++ mismatch: case {i}, round {r}")
    print(json.dumps({"official_cpp_random_cases":len(cases),"rounds":"1..28","status":"PASS"}))


if __name__=="__main__": main()
