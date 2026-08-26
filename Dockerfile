FROM python:3.12.13-slim-bookworm@sha256:a5d9a95a366e9cb09c32e2623ae98320433f169b2974b451969459ca585e009a

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /opt/cpg

COPY requirements-reviewer-lock.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip==26.2.1 \
    && python -m pip install --no-cache-dir -r requirements-reviewer-lock.txt

COPY . .

RUN python reviewer_verify.py \
    && python -B -m unittest -v test_single_realization_v2_6_2.py

CMD ["python", "reviewer_verify.py"]
