0. Clone repository
```
git clone ....
```
1. create virtual environment
```
python -m venv .venv
.venv/Script/activate
```
2. install torch torchvision, torchaudio (GPU)
```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
3. install library
```pip install -r requirements.txt```
4. run 
```python detect_api.py```