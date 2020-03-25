# Microinfarcts
* Microinfarcts is a project for loacting the real location of beads inside the brain and use ANTs to align the brain into Allen atlas. 
After going through the whole process, you will be able to get 
* 1. A csv file indicating the number of micro infarcts located in different brain regions.
* 2. A opencv window showing the standard atlas, symmerically normalized brain images and the location of the micro infarcts. 
* ![opencv window](https://github.com/SilasiLab/microinfarcts/blob/master/pics/show.png)
* You will be able to ajust the transparency using the buttons `q(-)` and `e(+)`, and also position the image using the buttons `a (previous imgae)` and `d (next image)`.
* The reference atlas comes from Allen Atlas organization. You can find reference data on google drive link attached here:(https://drive.google.com/drive/folders/10MqL8BXkfRsjLgWRuoxJUuZzH9AuWIOe?usp=sharing)
* After downloading the reference file, you need to copy it into `atlas_reference` folder.
* So the whole structure of the project should be:
    * `microinfarcts/src`
    * `microinfarcts/atlas_reference`
    * `microinfarcts/atlas_reference/annotation.mhd`
    * `microinfarcts/atlas_reference/annotation.raw`
    * `microinfarcts/atlas_reference/atlasVolume.mhd`
    * `microinfarcts/atlas_reference/atlasVolume.raw`

## 1. Install dependencies
 * 1. `conda install pandas`
 * 2. `conda install -c conda-forge ffmpeg`
 * 3. `conda install -c conda-forge opencv`
 * 4. `conda install matplotlib`
 * 5. `conda install pickle`
 * 6. `conda install tqdm`
 * 7. `conda install scikit-image`
 * 8. `pip install nipype`
 * 9. `conda install pyqt5`
 * 10. `conda install tk`
 * 11. Download and compile ANTs from (https://brianavants.wordpress.com/2012/04/13/updated-ants-compile-instructions-april-12-2012/)
 * 12. `git clone https://github.com/SilasiLab/microinfarcts.git`

## 2. Preparatory phase
  * 1. For the input raw data, the input folder directory structure should be: `[Your folder containing all the brains]/[brain id](individual brain)/raw/[images_b.jpg]`. Images should all have a postfix `b` (`imageid_b`, e.g.) which indicates color channel blue.
  * 2. You will need a folder to save the result as also. Feel free to create your own folders.
  * After downloading as well as compiling ANTs, you should find the dirctory of `antsRegistrationSyNQuick.sh` under ANTs `Scripts` folder. For an instance, it is `/home/username/ANTs/Scripts/antsRegistrationSyNQuick.sh`. As for the folder containing `antsRegistrationSyNQuick.sh`, that is `/home/username/ANTs/Scripts/` which will be used as the parameter `--ant` of the whole project. Here we leave it as [Script folder] for short and for future use.
  
## 3. User guide
  * 1. `cd [your directory]/microinfarcts/src`
  * 2. `python main.py`
  * 3. 
       ![Gui](/pics/microinfarctsGUI.png)
  * 4. Check `Auto Segmentation` will perform a Discrete Fourier Transform to find the microinfarcts(bright circle area) in brain scans. You can also manually segement the bead by selecting `Manual beads labeling`. Note: By running any one in the two methods, a csv file will be cached on the local disk. Once the file is cached, the programme will skip the step of segmentation.  
  * 5. Check `Show the result`, the script will transform the bead location mask and project it on the brain scan. Showing the result and generating the summary files have some conflicts, so they cannot be run at the same time.
  * 6. Check `Write a summary`, a summary csv, a structual tree txt, as well as a structual tree image will be generated under `[result folder]/[brain id]/`.
  * 7. Select the brain folder you would like to analyse from the left list, then move it to the analysing list on the right side by clicking the button `>>`.
  * 8. Click on the analyse button.
  * 9. A manual alignment is required at the begining. You need firstly click the common point shared by the two neigbour images. Go the last section containing the anterior commissure, draw the line from top to bottom in the right middle of the brain. Save it, then draw the line across the anterior commissure. Press Q to save all and continue the processing step.

## Note:
* For each step, the middle result is preserved in the local files and will be resued. So whenever you want to redo the manual work, you need to delete the relative files first.
* 1. The manual allign result is under `[Input folder]/[brain id]/[processed]`
* 2. The human labeled microinfarcts result is under `[Input folder]/[brain id]/[data]/manual_segmentation.csv`
* 3. The automatic labeled microinfarcts result is under `[Input folder]/[brain id]/[data]/auto_segmentation.csv`
* 4. Final result folder under `[result folder]/[brain id]` needs to be delected too.
