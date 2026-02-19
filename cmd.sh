
!python main.py --input data/input.mp4 --stabilize --model_size small

!python main.py --input data/input.mp4 --model_type depthanything --inpaint

!python main.py --input data/input.mp4 --model_type depthanything --inpaint --stabilize

!python main.py --input data/input.mp4 --mode heatmap --stabilize


!python main.py --input data/input.mp4 --mode sbs --stabilize --stabilizer_type optical_flow --inpaint

!python main.py --input data/input.mp4 --mode heatmap --stabilize --stabilizer_type optical_flow --inpaint

!python main.py --input data/input.mp4 --mode vr180 --stabilize --inpaint

!python main.py --input data/input.mp4 --mode vr360 --stabilize --inpaint

python main.py --input data/input.mp4 --mode anaglyph --stabilize --inpaint