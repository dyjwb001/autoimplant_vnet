import nrrd

for i in range(110):
    img,_=nrrd.read("D:\\Portal\\MedicalData\\Implant\\results\\resized_completeskull\\"+str(i).zfill(3)+".nrrd")
    nrrd.write("D:\\Portal\\MedicalData\\Implant\\results\\jianning\\"+str(i).zfill(3)+".nrrd",img)
