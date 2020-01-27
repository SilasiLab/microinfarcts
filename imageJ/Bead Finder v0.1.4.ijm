//********SETUP********
//Initial setup screen to prompt user to select relevant steps.
stackcheck=false;
rawcheck=false;
refcheck=false;
aligncheck=false;
reorientcheck=false;
beadcheck=false;
beadcountcheck=false;
beadcountanalysischeck=false;
scalecheck=false;
firstrun=true; 
slicecheck=false;
columnset=1;
prevnum=0;
numcheck=false;
firstprint=true;
defID="xxxxxxxx";
defX=0.0054645;
defY=0.0054645;
defZ=0.05;
defUnit="mm";
dilationfactor=0;

sH = screenHeight;
sW = screenWidth;

//Create dialog box to get user options.
Dialog.create("Which steps would you like to perform today?");
Dialog.addMessage("***PRE-ILASTIK PROCESSING***");
Dialog.addCheckbox("Create image stacks from raw images?", stackcheck);
Dialog.addCheckbox("Orient images?", rawcheck);
Dialog.addCheckbox("Do you have a reference image to display during image orientation?", refcheck);
Dialog.addCheckbox("Align images in stepwise pairs?", aligncheck);
Dialog.addCheckbox("Re-orient image stacks after alignment?", reorientcheck);
Dialog.addCheckbox("Manually label beads to serve as control for Ilastik?", beadcountcheck);
Dialog.addCheckbox("Analyze manually labelled bead ROI sets?", beadcountanalysischeck);
Dialog.addMessage("***POST-ILASTIK PROCESSING***");
Dialog.addCheckbox("Process Ilastik segmentations to obtain infarct volume/location?", beadcheck);
Dialog.show();
stackcheck = Dialog.getCheckbox();
rawcheck = Dialog.getCheckbox();;
refcheck = Dialog.getCheckbox();;;
aligncheck = Dialog.getCheckbox();;;;
reorientcheck = Dialog.getCheckbox();;;;;
beadcountcheck = Dialog.getCheckbox();;;;;;
beadcountanalysischeck = Dialog.getCheckbox();;;;;;;
beadcheck = Dialog.getCheckbox();;;;;;;;

//Custom function to allow easy appending of values to the end of an array (used later in script).
function append(arr, value) {
	arr2 = newArray(arr.length+1);
	for (i=0; i<arr.length; i++)
		arr2[i] = arr[i];
		arr2[arr.length] = value;
		return arr2;
}	

//Select directory where image files are located.
dir = getDirectory("Choose source directory");
list = getFileList(dir);

//Check to see if necessary sub-folders exist; if not, create them.

if (File.exists(dir+"1 - Unprocessed Raw Images")) {
} else {
	File.makeDirectory(dir+"1 - Unprocessed Raw Images");	
	}
stackdir = dir+"1 - Unprocessed Raw Images"+File.separator;

if (File.exists(dir+"3 - Processed Images")) {
} else {
	File.makeDirectory(dir+"3 - Processed Images");	
	}
imagedir = dir+"3 - Processed Images"+File.separator;

if (File.exists(imagedir+"1 - Processed Raw Images")) {
} else {
	File.makeDirectory(imagedir+"1 - Processed Raw Images");	
	}
procrawdir = imagedir+"1 - Processed Raw Images"+File.separator;

if (File.exists(imagedir+"2 - Raw Stacks")) {
} else {
	File.makeDirectory(imagedir+"2 - Raw Stacks");
	}
rawdir = imagedir+"2 - Raw Stacks"+File.separator;
rawlist = getFileList(rawdir);

if (rawcheck == true) {
	if (File.exists(dir+"Oriented Stacks")) {
	} else {
		File.makeDirectory(dir+"Oriented Stacks");	
		}

	if (File.exists(dir+"2 - Reference")) {
	} else {
		File.makeDirectory(dir+"2 - Reference");	
		}
	refdir = dir+"2 - Reference"+File.separator;
	reflist = getFileList(refdir);		
	oridir = dir+"Oriented Stacks"+File.separator;
}

if (reorientcheck == true) {
	if (File.exists(dir+"Oriented Stacks")) {
	} else {
		File.makeDirectory(dir+"Oriented Stacks");	
		}

	if (File.exists(dir+"2 - Reference")) {
	} else {
		File.makeDirectory(dir+"2 - Reference");	
		}
	refdir = dir+"2 - Reference"+File.separator;
	reflist = getFileList(refdir);		
	oridir = dir+"Oriented Stacks"+File.separator;
}

if (File.exists(imagedir+"3 - Unfinished Oriented Stacks Renamed")) {
} else {
	File.makeDirectory(imagedir+"3 - Unfinished Oriented Stacks Renamed");	
	}
oridirre = imagedir+"3 - Unfinished Oriented Stacks Renamed"+File.separator;

if (File.exists(imagedir+"4 - Finished Oriented Stacks Renamed")) {
} else {
	File.makeDirectory(imagedir+"4 - Finished Oriented Stacks Renamed");	
	}
finoridirre = imagedir+"4 - Finished Oriented Stacks Renamed"+File.separator;

if (File.exists(imagedir+"5 - Uncounted Aligned Stacks Renamed")) {
} else {
	File.makeDirectory(imagedir+"5 - Uncounted Aligned Stacks Renamed");	
	}
alidirre = imagedir+"5 - Uncounted Aligned Stacks Renamed"+File.separator;

if (File.exists(imagedir+"6 - Uncounted Reoriented Stacks Renamed")) {
} else {
	File.makeDirectory(imagedir+"6 - Uncounted Reoriented Stacks Renamed");	
	}
reorientdir = imagedir+"6 - Uncounted Reoriented Stacks Renamed"+File.separator;

if (File.exists(imagedir+"7 - Counted Reoriented Stacks Renamed")) {
} else {
	File.makeDirectory(imagedir+"7 - Counted Reoriented Stacks Renamed");	
	}
counteddir = imagedir+"7 - Counted Reoriented Stacks Renamed"+File.separator;

if (File.exists(dir+"5 - Data")) {
} else {
	File.makeDirectory(dir+"5 - Data");	
	}
datadir = dir+"5 - Data"+File.separator;

if (File.exists(imagedir+"8 - Bead Location")) {
} else {
	File.makeDirectory(imagedir+"8 - Bead Location");	
	}
beaddir = imagedir+"8 - Bead Location"+File.separator;

if (File.exists(imagedir+"9 - Segmentation")) {
} else {
	File.makeDirectory(imagedir+"9 - Segmentation");	
	}
segdir = imagedir+"9 - Segmentation"+File.separator;

if (File.exists(dir+"4 - ROIs")) {
} else {
	File.makeDirectory(dir+"4 - ROIs");	
	}
roidir = dir+"4 - ROIs"+File.separator;

run("Set Measurements...", "area mean min display redirect=None decimal=3");

//********CREATE IMAGE STACKS OUT OF RAW IMAGES********
//For-if statement to loop through all files in "1 - unprocessed raw images" and create initial stacks
if (stackcheck == true) {
	waitForUser("Ensure that images for only one animal are in Raw Images folder at the following directory:" + "\n" + " " + "\n" + stackdir);
	stacklist = getFileList(stackdir);
	Array.sort(stacklist);
	Dialog.create("Image stack file properties");
	Dialog.addMessage("***ENTER FILE PROPERTY INFORMATION AND PRESS OKAY TO CONTINUE***");
	Dialog.addString("File Name", defID, 8);
	Dialog.addNumber("Pixel Width (X)", defX, 7, 9, "");
	Dialog.addNumber("Pixel Height (Y)", defY, 7, 9, "");	
	Dialog.addNumber("Voxel Depth (Z)", defZ, 3, 5, "");
	Dialog.addString("Measument Unit", defUnit, 8);
	Dialog.show();
	imgName = Dialog.getString();
	width = Dialog.getNumber();
	height = Dialog.getNumber();;
	depth = Dialog.getNumber();;;
	unit = Dialog.getString();;
	for (i=0; i<stacklist.length; i++) { 
		if (endsWith(stacklist[i], ".jpg")) {	
			open(stackdir+stacklist[i]);
		}
	}
	run("Images to Stack", "name=Stack title=[]");
	run("Split Channels");
	selectWindow("Stack (red)");
	run("Close");
	selectWindow("Stack (blue)");
	run("Close");
	selectWindow("Stack (green)");
	rename(imgName);
	setVoxelSize(width, height, depth, unit);
	saveAs("Tiff", rawdir + "_" + imgName + ", ");
	run("Close All");	
	for (i=0; i<stacklist.length; i++) {
		File.copy(stackdir+stacklist[i], procrawdir+stacklist[i]);
		File.delete(stackdir+stacklist[i]);
	}
	selectWindow("Log");
	run("Close");
}

//********ORIENTATION AND RESAVING OF STACKS FOR ALL FUTURE PROCESSING********
//For-if statement to loop through all files in "2 - raw stacks" directory and re-align them to common RAS-orientation
if (rawcheck == true) {
	for (i=0; i<rawlist.length; i++) { 
		if (endsWith(rawlist[i], ".tif")) {
			setTool("line");
			open(rawdir+rawlist[i]);
			imgName=getTitle();
			imgNameindex = indexOf(imgName, ".");
			imgNamesub = substring(imgName, 1, imgNameindex);
		    numstart = indexOf(imgName, "_");
		    numend = indexOf(imgName, ", ");
		    num = substring(imgName, numstart+1, numend);
			totslice = nSlices;
			getVoxelSize(width,height,depth,unit);
			newname = "_" + imgNamesub + "x" + width + ", y" + height + ", z" + depth + ", u" + unit + ", .";
			rename(newname);
			run("Maximize");
			if (refcheck == true) {
				open(refdir+reflist[0]);
				refname=getTitle();
				selectWindow(newname);
				setLocation(0,0);
				selectWindow(refname);
				setLocation((sW/2),0);
				selectWindow(newname);
			}
			if (slicecheck == false) {
				waitForUser("Set the slice to the most posterior section where the anterior commisure crosses the midline and press okay");
				targslice = getSliceNumber();
				Dialog.create("Target slice setting");
				Dialog.addCheckbox("Would you like to use this slice for other images in this macro run?", slicecheck);
				Dialog.show();
				slicecheck = Dialog.getCheckbox();	
			}
			sliceset = totslice*(targslice/totslice);
			setSlice((sliceset));
		    waitForUser("Draw a vertical line (top-down) through the midline of the brain on any section and press okay");
		    run("Measure");
		    angle = getResult("Angle");
		    corrangle = angle+90;
		    selectWindow("Results");
			run("Close");
		    run("Rotate... ", "angle=corrangle grid=1 interpolation=Bilinear stack");
		    run("Select None");
		    setTool("point");
		   	waitForUser("Click a point on the center of the anterior commisure on the most posterior section where it crosses the midline");
		   	run("Measure");
		   	xpoint = getResult("X")/width;
		   	ypoint = getResult("Y")/height;
		   	targetslice = getResult("Slice");
		   	mmscale = depth;
			selectWindow("Results");
			run("Close");
			getDimensions(width, height, channels, slices, frames);
			xtranslate = (width/2) - xpoint;
			ytranslate = (height/2) - ypoint;
			run("Translate...", "x=xtranslate y=ytranslate interpolation=None stack");		   	
			selectWindow(newname);
			run("Select None");
			run("Image Sequence... ", "format=TIFF start=1 digits=3 save=oridir");
			run("Close All");

//Rename each file with AP coordinate relative to the crossing of the anterior commisure
			oridirlist = getFileList(oridir);
			for(j=0; j<oridirlist.length; j++) {
				if (indexOf(oridirlist[j],num)>=1) {
					open(oridir + oridirlist[j]);
					subimgName=getTitle();
					substart = lastIndexOf(subimgName, ", ");
					subend = indexOf(subimgName, ".tif");
					subnum = substring(subimgName, substart+1, subend);
					subnumFloat = parseFloat(subnum);
					APcoord = (targetslice - subnumFloat) * mmscale;
					subimgNameindex = lastIndexOf(subimgName, ".");
					subimgNamesub = substring(subimgName, 0, subimgNameindex);
					newsubimgName = subimgNamesub + ", " + APcoord + ".tif";
					rename(newsubimgName);
					saveAs("Tiff", oridirre + newsubimgName);
					run("Close All");
				}
			}
		}
	}

//Clean up temporary directories for image orientation
	if(File.exists(oridir)) {
		oridirlist = getFileList(oridir);
		for(k=0; k<oridirlist.length; k++) {
			File.delete(oridir + oridirlist[k]);
		}
		File.delete(oridir);
		selectWindow("Log");
		run("Close");
	}
}

//********ALIGNMENT AND RESAVING OF ORIENTED IMAGES FOR ALL FUTURE PROCESSING********
//For-if statement to loop through all files in directory, "3 - unfinished oriented renamed", in pairs and align image 2 to image 1
if (aligncheck == true) {
	run("Set Measurements...", "area centroid display redirect=None decimal=3");
	oridirrelist = getFileList(oridirre);
	Array.sort(oridirrelist);
	for (n=0; n<oridirrelist.length; n++) {
		if (n!=oridirrelist.length-1) {
			if (endsWith(oridirrelist[n], ".tif")) {
				alidirrelist = getFileList(alidirre);
				if(File.exists(alidirre + oridirrelist[n])) {
					open(alidirre + oridirrelist[n]);			
				} else {
					File.copy(oridirre+oridirrelist[n], alidirre+oridirrelist[n]);					
					open(alidirre + oridirrelist[n]);
				}
				getVoxelSize(width,height,depth,unit);
				imgName=getTitle();
			    numstart1 = indexOf(imgName, "_");
			    numend1 = indexOf(imgName, ", ");
			    num1 = substring(imgName, numstart1+1, numend1);
				run("Maximize");
				selectWindow(imgName);
				setLocation(0,0);
				open(oridirre + oridirrelist[n+1]);
				imgName2=getTitle();
			    numstart2 = indexOf(imgName2, "_");
			    numend2 = indexOf(imgName2, ", ");
			    num2 = substring(imgName2, numstart2+1, numend2);
				if (num1 == num2) {
					run("Maximize");
					selectWindow(imgName2);
					setLocation((sW/2),0);
					setTool("multipoint");
					run("Point Tool...", "type=Dot color=Black size=Small counter=0");
					waitForUser("Click a common bead that appears in both the left and right images, then press Okay.");
					selectWindow(imgName);
					run("Measure");
					selectWindow(imgName2);
					run("Measure");
					x0 = getResult("X", 0)/width;
					x1 = getResult("X", 1)/width;
					xdiff = x0-x1;
					y0 = getResult("Y", 0)/height;
					y1 = getResult("Y", 1)/height;
					ydiff = y0-y1;
					selectWindow(imgName2);		
					run("Translate...", "x=xdiff y=ydiff interpolation=None");
					selectWindow("Results");
					run("Close");
					selectWindow(imgName2);	
					run("Select None");
					saveAs("Tiff", alidirre + imgName2);
					run("Close All");	
					File.copy(oridirre+oridirrelist[n], finoridirre+oridirrelist[n]);
					File.delete(oridirre+oridirrelist[n]);
					selectWindow("Log");
					run("Close");
				} else {
					run("Close All");	
					File.copy(oridirre+oridirrelist[n], finoridirre+oridirrelist[n]);
					File.delete(oridirre+oridirrelist[n]);
					selectWindow("Log");
					run("Close");
				}
			}		
		} else {
			File.copy(oridirre+oridirrelist[n], finoridirre+oridirrelist[n]);
			File.delete(oridirre+oridirrelist[n]);
			selectWindow("Log");
			run("Close");
		}
	}
}

//********RE-ORIENTATION AND STACK FLIPPING OF ALIGNED IMAGES********
//For-if statement to loop through all files in "-1 - Unprocessed raw" directory, reorient and flip the stack.
if (reorientcheck == true) {
	waitForUser("Ensure that images for only one animal are in Raw Images folder at the following directory:" + "\n" + " " + "\n" + stackdir);
	stacklist = getFileList(stackdir);
	Array.sort(stacklist);
	Dialog.create("Image stack file properties");
	Dialog.addMessage("***ENTER FILE PROPERTY INFORMATION AND PRESS OKAY TO CONTINUE***");
	Dialog.addString("File Name", defID, 8);
	Dialog.addNumber("Pixel Width (X)", defX, 7, 9, "");
	Dialog.addNumber("Pixel Height (Y)", defY, 7, 9, "");	
	Dialog.addNumber("Voxel Depth (Z)", defZ, 3, 5, "");
	Dialog.addString("Measument Unit", defUnit, 8);
	Dialog.show();
	imgName = Dialog.getString();
	width = Dialog.getNumber();
	height = Dialog.getNumber();;
	depth = Dialog.getNumber();;;
	unit = Dialog.getString();;
	for (i=0; i<stacklist.length; i++) { 
		if (endsWith(stacklist[i], ".tif")) {	
			open(stackdir+stacklist[i]);
		}
	}
	run("Images to Stack", "name=Stack title=[]");
	rename(imgName);
	setVoxelSize(width, height, depth, unit);
	run("Flip Z");
	setTool("line");
	totslice = nSlices;
	newname = "_" + imgName + ", x" + width + ", y" + height + ", z" + depth + ", u" + unit + ", .";
	rename(newname);
	run("Maximize");
	if (refcheck == true) {
		open(refdir+reflist[0]);
		refname=getTitle();
		selectWindow(newname);
		setLocation(0,0);
		selectWindow(refname);
		setLocation((sW/2),0);
		selectWindow(newname);
	}
	if (slicecheck == false) {
		waitForUser("Set the slice to the most posterior section where the anterior commisure crosses the midline and press okay");
		targslice = getSliceNumber();
		Dialog.create("Target slice setting");
		Dialog.addCheckbox("Would you like to use this slice for other images in this macro run?", slicecheck);
		Dialog.show();
		slicecheck = Dialog.getCheckbox();	
	}
	sliceset = totslice*(targslice/totslice);
	setSlice((sliceset));
    waitForUser("Draw a vertical line (top-down) through the midline of the brain on any section and press okay");
    run("Measure");
    angle = getResult("Angle");
    corrangle = angle+90;
    selectWindow("Results");
	run("Close");
    run("Rotate... ", "angle=corrangle grid=1 interpolation=Bilinear stack");
    run("Select None");
    setTool("point");
   	waitForUser("Click a point on the center of the anterior commisure on the most posterior section where it crosses the midline");
   	run("Measure");
   	xpoint = getResult("X")/width;
   	ypoint = getResult("Y")/height;
   	targetslice = getResult("Slice");
   	mmscale = depth;
	selectWindow("Results");
	run("Close");
	getDimensions(width, height, channels, slices, frames);
	xtranslate = (width/2) - xpoint;
	ytranslate = (height/2) - ypoint;
	run("Translate...", "x=xtranslate y=ytranslate interpolation=None stack");		   	
	selectWindow(newname);
	run("Select None");
	run("Image Sequence... ", "format=TIFF start=1 digits=3 save=oridir");
	run("Close All");

//Rename each file with AP coordinate relative to the crossing of the anterior commisure
	oridirlist = getFileList(oridir);
	for(j=0; j<oridirlist.length; j++) {
		if (indexOf(oridirlist[j],imgName)>=1) {
			open(oridir + oridirlist[j]);
			subimgName=getTitle();
			substart = lastIndexOf(subimgName, ", ");
			subend = indexOf(subimgName, ".tif");
			subnum = substring(subimgName, substart+1, subend);
			subnumFloat = parseFloat(subnum);
			APcoord = (subnumFloat - targetslice) * mmscale;
			subimgNameindex = lastIndexOf(subimgName, ".");
			subimgNamesub = substring(subimgName, 0, subimgNameindex);
			newsubimgName = subimgNamesub + ", " + APcoord + ".tif";
			rename(newsubimgName);
			saveAs("Tiff", reorientdir + newsubimgName);
			run("Close All");
		}
	}
	
//Clean up temporary directories for image orientation
	if(File.exists(oridir)) {
		oridirlist = getFileList(oridir);
		for(k=0; k<oridirlist.length; k++) {
			File.delete(oridir + oridirlist[k]);
		}
		File.delete(oridir);
		selectWindow("Log");
		run("Close");
	}
	run("Close All");	
	for (i=0; i<stacklist.length; i++) {
		File.copy(stackdir+stacklist[i], alidirre+stacklist[i]);
		File.delete(stackdir+stacklist[i]);
	}
	selectWindow("Log");
	run("Close");
}

//********HUMAN COUNTING ORIGINAL IMAGES TO SERVE AS CONTROL FOR ILASTIK IMAGES********
//Hand-labelling of beads on images that would be input into Ilastik
if (beadcountcheck == true) {
	run("Set Measurements...", "area centroid display redirect=None decimal=3");
	oridirrelist = getFileList(reorientdir);
	Array.sort(oridirrelist);
	bcc = false;
	for (n=0; n<oridirrelist.length; n++) {
		if (endsWith(oridirrelist[n], ".tif")) {
			open(reorientdir + oridirrelist[n]);
			run("Set Scale...", "distance=0 known=1 pixel=1 unit=um");
			imgName=getTitle();
			imgH = getHeight();
			imgW = getWidth();
			nindex = indexOf(imgName, "_");
			xindex = indexOf(imgName, "x");
			yindex = indexOf(imgName, "y");
			zindex = indexOf(imgName, "z");
			uindex = indexOf(imgName, "u");
			cindex = lastIndexOf(imgName, ", ");
			pindex = lastIndexOf(imgName, ".");
			num = substring(imgName, nindex+1, xindex-2);
			APcoord = substring(imgName, cindex+2, pindex);
			run("Maximize");
			setTool("multipoint");
			run("Point Tool...", "type=Dot color=Black size=Small counter=0");
			Dialog.create("Are microbeads present in the image?");
			Dialog.addCheckbox("Are microbeads present in this image?", bcc);
			Dialog.show();
			bcc = Dialog.getCheckbox();
			if (bcc == true) {
				waitForUser("Click all microbeads in image and press Okay");
				roiManager("Add");
				roiManager("Select", 0);
				roiManager("Rename", APcoord);
				roiManager("Deselect");
				roiManager("Save", roidir + imgName + ".zip");
				selectWindow("ROI Manager");
				run("Close");
			}
			run("Close All");
			File.copy(reorientdir+oridirrelist[n], counteddir+oridirrelist[n]);
			File.delete(reorientdir+oridirrelist[n]);
			selectWindow("Log");
			run("Close");
		}
	}	
}

//Analysis of ROIs from hand-labelling of beads
if (beadcountanalysischeck == true) {
	Dialog.create("Number of dilations in bead analysis");
	Dialog.addNumber("Enter dilation factor for adjacent beads?", dilationfactor, 0, 2, "");
	Dialog.addMessage("Note: Each unit of dilation factor will expand bead coordinates by ~10 um.");
	Dialog.show();
	dilationfactor = Dialog.getNumber();
	run("Set Measurements...", "area mean centroid shape limit display redirect=None decimal=3");
	roilist = getFileList(roidir);
	countedlist = getFileList(counteddir);
	Array.sort(countedlist);	
	beadcountarray = newArray(0);
	beadlocationarray = newArray(0);
	dataarray = newArray(0);
	numarray = newArray(0);
	truebeadcountarray = newArray(0);	
	dataentry = "";
	beadlocationdata = ""; 
	for (n=0; n<countedlist.length; n++) {
		if (endsWith(countedlist[n], ".tif")) {
			nindex = indexOf(countedlist[n], "_");
			xindex = indexOf(countedlist[n], "x");
			num = substring(countedlist[n], nindex+1, xindex-2);
			if (firstrun == true) {
				prevnum = num;
				numarray = append(numarray, num);
				firstrun = false;
			}
			if (prevnum != num) {
				numarray = append(numarray, num);
			}
		}
	}		
	for(n=0; n<numarray.length; n++) {
		for (o=0; o<roilist.length; o++) {
			if (indexOf(roilist[o],numarray[n])>=0) {
				imgIndex = indexOf(roilist[o], ".zip");
				nindex = indexOf(roilist[o], "_");
				xindex = indexOf(roilist[o], "x");
				cindex = lastIndexOf(roilist[o], ", ");
				pindex = lastIndexOf(roilist[o], ".");
				APcoord = substring(roilist[o], cindex+2, pindex-4);
				num = substring(roilist[o], nindex+1, xindex-2);
				imgName = substring(roilist[o],0,imgIndex);
				open(counteddir + imgName);
				getVoxelSize(width,height,depth,unit);
				imgH = getHeight();
				imgW = getWidth();
				open(roidir + roilist[o]);
				roiManager("Select", o);
				run("Measure");
				beadcount = nResults();
				beadcountarray = append(beadcountarray, beadcount);
				if(beadcount > 0) {
					for(j=0; j<nResults; j++) {
						xcoord = getResult("X", j);
						xcoordcalc = ((imgW/2)*width) - xcoord; 
						ycoord = getResult("Y", j);
						ycoordcalc = ((imgH/2)*height) - ycoord;
						beadarea = getResult("Area", j);
						beadcircularity = getResult("Circ.", j);
						beadlocationdata = numarray[n] + ", " + xcoordcalc + ", " + ycoordcalc + ", " + APcoord + ", " + beadarea + ", " + beadcircularity;
						beadlocationarray = append(beadlocationarray, beadlocationdata);				
					}
				selectWindow("Results");
				run("Close");
				}
			}				 	
		}	
		run("Images to Stack", "name=Stack title=[] use");
		run("Z Project...", "projection=[Average Intensity]");
		selectWindow("Stack");
		run("Close");
		selectWindow("AVG_Stack");
		rename(numarray[n]);
		run("Select All");
		setBackgroundColor(255, 255, 255);
		run("Clear", "slice");
		run("Select None");
		setForegroundColor(0, 0, 0);
		ROInumber = roiManager("count");
		for (p=0; p<ROInumber; p++) {
			roiManager("Select", p);			
			run("Measure");
		}
		for (q=0; q<nResults(); q++) {
			xcoord = getResult("X", q);			
			ycoord = getResult("Y", q);
			drawLine(xcoord/width,ycoord/width,xcoord/width,ycoord/width);
		}
		selectWindow("Results");
		run("Close");

		run("Select None");
		if (dilationfactor != 0) {
			for (d=0; d<dilationfactor; d++) {
				run("Dilate");
			}
		}
		run("Analyze Particles...", "display add");	
		selectWindow("Results");
		truebeadcount = nResults();
		truebeadcountarray = append(truebeadcountarray, truebeadcount);
		if(truebeadcount > 0) {
			for(j=0; j<nResults; j++) {
				APcoord = "Flat";
				xcoord = getResult("X", j);
				xcoordcalc = ((imgW/2)*width) - xcoord; 
				ycoord = getResult("Y", j);
				ycoordcalc = ((imgH/2)*height) - ycoord;
				beadarea = getResult("Area", j);
				beadcircularity = getResult("Circ.", j);
				beadlocationdata = numarray[n] + ", " + xcoordcalc + ", " + ycoordcalc + ", " + APcoord + ", " + beadarea + ", " + beadcircularity;
				beadlocationarray = append(beadlocationarray, beadlocationdata);				
			}
		selectWindow("Results");
		run("Close");
		}	
		beadcountsum=0;
		for (m=0; m<beadcountarray.length; m++) {
			beadcountsum = beadcountsum + beadcountarray[m];
		}
		dataentry = num + ", " + beadcountsum + ", " + truebeadcount;
		dataarray = append(dataarray, dataentry);
		beadcountarray = newArray(0);
		selectWindow(numarray[n]);
		run("Select None");
		flatname = numarray[n] + " - Flattened Bead Locations.tif";
		saveAs("Tiff", beaddir + flatname);							
		run("Close All");
		print("Animal ID, X, Y, Z, Bead Area, Bead Circularity");
		for(m=0; m<beadlocationarray.length; m++) {
			print(beadlocationarray[m]);
		}
		selectWindow("Log");
		beadname = "Manual Bead Location Data v0.1.4 - Dilation Factor " + dilationfactor + ".csv";
		saveAs("Text", datadir+beadname);
		selectWindow("Log");
		run("Close");
		selectWindow("ROI Manager");
		run("Close");
	}
	print("Animal ID, Total Bead Count, Flattened Bead Count");
	for(m=0; m<dataarray.length; m++) {
		print(dataarray[m]);
	}
	selectWindow("Log");
	manualbeadfile = "Manual bead counting v0.1.4 - Dilation Factor " + dilationfactor + ".csv";
	saveAs("Text", datadir+manualbeadfile);
	selectWindow("Log");
	run("Close");
}

//***PROCESS POST-ILASTIK SEGMENTATIONS OF INFARCT VOLUME AND LOCATION RELATIVE TO ANTERIOR COMMISURE***
//Open images in sequence and calibrate scale using information in file name
if (beadcheck == true) {
	run("Set Measurements...", "area centroid shape limit display redirect=None decimal=3");
	segdirlist = getFileList(segdir);
	countarray = newArray(0);
	concatareaarray = newArray(0);
	concatbasearray = newArray(0);
	APcoordarray = newArray(0);
	blankarray = newArray(0);
	for (i=0; i<segdirlist.length; i++) {
		open(segdir+segdirlist[i]);
		basearray = newArray(0);
		areaarray = newArray(0);
		imgName=getTitle();
		imgH = getHeight();
		imgW = getWidth();
		nindex = indexOf(imgName, "_");
		xindex = indexOf(imgName, "x");
		yindex = indexOf(imgName, "y");
		zindex = indexOf(imgName, "z");
		uindex = indexOf(imgName, "u");
		cindex = lastIndexOf(imgName, ", ");
		pindex = lastIndexOf(imgName, ".");
		num = substring(imgName, nindex+1, xindex-2);
	
//Check if image is from new animal and save output as a .csv log file if new animal is being analyzed	
		if (firstrun == true) {
			prevnum = num;
			print("X, Y, Z, Bead Area, Bead Circularity");
			firstrun = false;
		}
		if(prevnum != num) {
			selectWindow("Log");
			beadname = prevnum + " - Individual Bead Count Data" + ".csv";
			saveAs("Text", datadir+beadname);
			selectWindow("Log");
			run("Close");
			print("X, Y, Z, Bead Area, Bead Circularity");
		}

//Continue previous analysis of image after image identity has been checked.							
		xscale = substring(imgName, xindex+1, yindex-2);
		xscale = parseFloat(xscale);
		yscale = substring(imgName, yindex+1, zindex-2);
		yscale = parseFloat(yscale);
		zscale = substring(imgName, zindex+1, uindex-2);
		zscale = parseFloat(zscale);
		uscale = substring(imgName, uindex+1, cindex-4);
		APcoord = substring(imgName, cindex+2, pindex);
		setVoxelSize(xscale, yscale, zscale, uscale);
		rename(APcoord);
		setThreshold(2, 255);
		run("Convert to Mask");
		run("Analyze Particles...", "display summarize");
		selectWindow("Summary");
		lines = split(getInfo(), "\n");
		columns = split(lines[1], "\t");
		count = columns[1];
		count = parseFloat(count);
		if (prevnum == num) {
			countarray = append(countarray, count);
			APcoordarray = append(APcoordarray, APcoord);
		}
		selectWindow("Summary");
		run("Close");		
		if(count > 0) {
			for(j=0; j<nResults; j++) {
				xcoord = getResult("X", j);
				xcoordcalc = (imgW/2) - xcoord; 
				ycoord = getResult("Y", j);
				ycoordcalc = (imgH/2) - ycoord;
				beadarea = getResult("Area", j);
				beadcircularity = getResult("Circ.", j);
				print(xcoordcalc + ", " + ycoordcalc + ", " + APcoord + ", " + beadarea + ", " + beadcircularity);
			}
			selectWindow("Results");
			run("Close");
		}
		prevnum = num;
		run("Close All");
	}

//Save output of final animal as a .csv log file	
	selectWindow("Log");
	beadname = num + " - Individual Bead Count Data" + ".csv";
	saveAs("Text", datadir+beadname);
	selectWindow("Log");
	run("Close");
}