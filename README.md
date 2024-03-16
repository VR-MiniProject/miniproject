# Part 3d

Output videos : https://drive.google.com/drive/folders/1D3iN2I596R53vvwNel3u-8bbp18-Mg8b?usp=sharing

## How to run
### YOLO-V5 
Navigate to the corresponding folder and you will find the readme file there showing how to run the project. Skip the weight downloading part as the fine tuned model's weight is already stored in the weight folder and the command is already present in main.py. 
After installing all the required libraries, you can simply run using 
```
python main.py --input_path junction.mp4
```
This will store the ouput in output folder

### Faster RCNN
Navigate to the corresponding folder. This uses the same base library so the readme will be same as in the previous step, ignore that. You need to install all the same libraries that you did in previous step and along with it, you need to download detectron2. 
Note : You may need different pytorch and cuda version to succesfully install detectron2. After this, run the same command as above and you will have the output in output folder.
