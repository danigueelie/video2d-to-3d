
!python main.py --input data/input.mp4 --stabilize --model_size small

!python main.py --input data/input.mp4 --model_type depthanything --inpaint

!python main.py --input data/input.mp4 --model_type depthanything --inpaint --stabilize

!python main.py --input data/input.mp4 --mode heatmap --stabilize


!python main.py --input data/input.mp4 --mode sbs --stabilize --stabilizer_type optical_flow --inpaint

!python main.py --input data/input.mp4 --mode heatmap --stabilize --stabilizer_type optical_flow --inpaint

!python main.py --input data/input.mp4 --mode vr180 --stabilize --inpaint

!python main.py --input data/input.mp4 --mode vr360 --stabilize --inpaint

python main.py --input data/input.mp4 --mode anaglyph --stabilize --inpaint

python main.py --input data/input.mp4 --mode heatmap_numbered --stabilize


****************
python main.py --input data/input.mp4 --mode heatmap_numbered --stabilize --save_frames

**Ce qui va se passer :**
Dans ton dossier `outputs/`, un nouveau dossier portant le nom de ta vidéo va être créé. À l'intérieur, tu trouveras :
* `frames/` : Les images originales de la vidéo (`frame_00001.png`, etc.)
* `depths/` : Les cartes de profondeur en noir et blanc (plus c'est clair, plus c'est proche).
* `outputs/` : Le résultat final de ton traitement (par exemple la heatmap ou le SBS 3D).

*Note : Attention à l'espace disque ! Extraire chaque frame d'une vidéo peut prendre beaucoup de place (1 minute de vidéo à 30fps = 1800 images par dossier).*