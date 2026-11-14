# syntax=docker/dockerfile:1
# Build only from the verified context emitted by prepare.py context.
FROM nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
COPY pins.json /build/pins.json
RUN python3 - <<'PY'
import hashlib,json,pathlib,tarfile,urllib.request
p=json.load(open('/build/pins.json'))['compiler']
a=pathlib.Path('/tmp/zig.tar.xz');urllib.request.urlretrieve(p['url'],a)
assert hashlib.sha256(a.read_bytes()).hexdigest()==p['archive_sha256']
with tarfile.open(a) as t:t.extractall('/opt/zig-extract',filter='data')
root=next(pathlib.Path('/opt/zig-extract').iterdir());root.rename('/opt/zig')
assert hashlib.sha256(pathlib.Path('/opt/zig/zig').read_bytes()).hexdigest()==p['binary_sha256']
PY
RUN git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar /xgrammar && \
    test "$(git -C /xgrammar rev-parse HEAD)" = 97787376faee5ed8466cfad57c99855e4ce2f6aa
COPY source/ /source/
WORKDIR /source
RUN /opt/zig/zig build -j2 dsv41-assets dsv41-engram-pack dsv41-serve grammar \
    -Dxgrammar=/xgrammar -Doptimize=ReleaseSafe -Dsm=121 \
    -Dnvcc=/usr/local/cuda/bin/nvcc -Ddsv41-nvcc=/usr/local/cuda/bin/nvcc \
    -Dversion=1.2.9-dev+mia-mainline.2ba55ec7 --prefix /opt/tensorfold-main && \
    test -x /opt/tensorfold-main/bin/tensorfold-dsv41 && \
    test -x /opt/tensorfold-main/bin/tf-dsv41 && \
    test -s /opt/tensorfold-main/native/lib/libtfgrammar.so
RUN mkdir -p /opt/tensorfold-main/share && \
    cp /source/LICENSE /source/NOTICE /source/THIRD_PARTY_NOTICES.md /opt/tensorfold-main/share/ && \
    cp /xgrammar/LICENSE /opt/tensorfold-main/share/xgrammar-LICENSE && \
    cp /xgrammar/NOTICE /opt/tensorfold-main/share/xgrammar-NOTICE && \
    cp /xgrammar/3rdparty/dlpack/LICENSE /opt/tensorfold-main/share/dlpack-LICENSE
LABEL tf.engine=mainline tf.ref=2ba55ec7d32d0bec446bb012678550be7ef52895 tf.mainline.qualification=untested
ENV PATH=/opt/tensorfold-main/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/tensorfold-main/native/lib:${LD_LIBRARY_PATH}
WORKDIR /workspace
# No server entrypoint: image construction never launches a model.
CMD ["tensorfold-dsv41", "--help"]
