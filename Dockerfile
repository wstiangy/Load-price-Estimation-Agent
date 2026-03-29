FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app/src \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR $HOME/app

COPY --chown=user requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user README.md pyproject.toml ./

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "lmp_agent.dashboard:app", "--host", "0.0.0.0", "--port", "7860"]
