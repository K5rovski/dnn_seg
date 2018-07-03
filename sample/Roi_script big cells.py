# ----------------------------------------------------------
#
# Run inside imagej
#
# ----------------------------------------------------------


from ij import IJ 
from ij.io import FileSaver 
import os 
from ij.plugin.frame import RoiManager
from ij.gui import Roi
from ij.measure import ResultsTable
from ij.plugin.filter import ParticleAnalyzer
from ij.plugin.filter.ParticleAnalyzer import ADD_TO_MANAGER,EXCLUDE_EDGE_PARTICLES
import time


lookup_path=r"D:\code_projects\Neuron_fibers_workspace\neuron_fiber_images\weka_predictions_interim"
saving_dir=r"D:\code_projects\Neuron_fibers_workspace\neuron_fiber_images\sample_imgs\saved_rois_bigcells_2k_weka"

doFileSave=True
doRmClose=False
leaveMasks=False

train_images=('sp13909-img01,sp13909-img02,sp13938-img07,'+
'sp13726-img01,sp14252-img05,sp13909-img07,sp13909-img08,'+
'sp13909-img03,sp13933-img04,sp13726-img04,sp13726-img03,'+
'sp13933-img08,sp13909-img05,sp13726-img02,sp13933-img05,'+
'sp13933-img06,sp13933-img07,sp13909-img06,sp13726-img08').split(',')


proc_images=[os.path.join(lookup_path,filename) for ind,filename in enumerate(os.listdir(lookup_path) )  if  filename[:filename.rindex('img')+5] in train_images ]
#print(len(proc_images))


if not os.path.exists(saving_dir):
	os.makedirs(saving_dir)
	print('i made it')
else:
	print('it already exists')



rm=None

for img_ind,image in enumerate(proc_images):
	imp = IJ.openImage(image);
	imp.show()

	IJ.setThreshold(128,128)

	#rm = None#RoiManager.getInstance()
	if not rm:
	  rm = RoiManager()
	else:
		rm.reset()
	
	#IJ.run(imp,"Analyze Particles...", "size=112-Infinity exclude add")
	rt=ResultsTable()
	pa=ParticleAnalyzer(EXCLUDE_EDGE_PARTICLES | ADD_TO_MANAGER,
                        0,
                        rt,
                        200,
                        float("inf"),
                        0.0,
                        1.0)

	pa.analyze(imp)
	# time.sleep(2)
	print 'Size of results: ',rt.size()
	# rm.runCommand("select","all")
	# rm.runCommand("Fill","3")
	save_path=saving_dir+"\\mask_%s" % (image[image.rindex('\\')+1:])
	# print(save_path)
	impMask = IJ.createImage("Mask", "8-bit grayscale-mode", imp.getWidth(), imp.getHeight(), imp.getNChannels(), imp.getNSlices(), imp.getNFrames())
	impMask.show()
	IJ.setForegroundColor(255, 255, 255)
	
	rm.runCommand(impMask,"Deselect")
	rm.runCommand(impMask,"Draw")
	

	if doFileSave and FileSaver(impMask).saveAsTiff(save_path):
		print 'Saved Image ',image
	else:
		print '!!! Not saved Image',image
	
	if not leaveMasks and img_ind<len(proc_images)-1:
		impMask.changes=False
		impMask.close()	
		imp.changes=False
		imp.close()
	
	# rm.runCommand("save selected", save_path)
if doRmClose: rm.close()