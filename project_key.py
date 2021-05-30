import numpy as np
from matplotlib import pyplot as plt
import cv2
from random import random
import math
import imageio
import scipy.stats as si

class ChaoticProcess:
    def __init__(self,img,label):
        self.img=img
        self.original=img.copy()
        self.N=len(img)
        self.label=label
        self.halfLen=self.N**0.5

        
    def encrypt(self):
        self.img=self.Stuffing(self.img,self.N//2)
        self.showImage(self.img,self.label+'Shuffing')
        print('*'*30)

        
        self.ikeda_key=self.getIkedaMap(64*64,50,0.92,10,self.img)
        self.showImage(self.ikeda_key,self.label+'Ikeda')
        print('*'*30)

        
        self.henonKey=np.array(self.getHenonMap(256,0.4,0.4)).astype(np.uint8)
        # self.img=np.bitwise_xor(self.img,self.henonKey,self.ikeda_key)
        print('*'*30)
        self.showImage(self.henonKey,self.label+'Henon')

        self.showImage(self.img,self.label+'Encrypted')


        # for i in self.img:
        #     print(i)

        self.histogramAnalysis(self.original,self.img,self.label+': Original',self.label+': Encrypted')
        
        return self.img



    def decrypt(self):
        self.original=self.img.copy()
        
        self.ikeda_key=self.getIkedaMap(64*64,50,0.92,10,self.img)
        print('*'*30)
        self.showImage(np.bitwise_xor(self.img,self.ikeda_key),self.label+'Decrypt Ikeda')

        
        self.henonKey=np.array(self.getHenonMap(256,0.4,0.4)).astype(np.uint8)
        self.img=np.bitwise_xor(self.img,self.henonKey,self.ikeda_key)
        print('*'*30)
        self.showImage(self.henonKey,self.label+'Decrypt Henon')

        self.showImage(self.img,self.label+'Decrpt Exor')

        self.img=self.decrptionStuffing(self.img,self.N//2)
        print('*'*30)
        self.showImage(self.img,self.label+'Decrypted')

        # self.histogramAnalysis(self.original,self.img,self.label+': Encypted',self.label+': Decrypted')
        # self.sbox=self.generateSBox()
        # self.img=np.bitwise_xor(self.img,self.sbox)
        # self.showImage(self.img,'S-Box')

        return self.img


        

    def Stuffing(self,img,half_len):
        c=0
        
        for i in range(1,2):
            img=self.permuteRow(img,half_len)
            # img=np.rot90(img)
            img=self.permuteCol(img,half_len)
            # img=self.diagnolStuffing(img)
            # img=np.rot90(img,3)
            print(self.label+': Stuffing iteration...')

        for i in range(3):            
            img=self.diagnolStuffing(img,half_len*2)
            img=np.rot90(img)
        

        return img

    def permuteRow(self,img,half_index):
        # half_index=self.N//2
        for i in range(0,half_index,2):
            for j in range(half_index*2):
                img[i][j],img[half_index+i][j]=img[half_index+i][j],img[i][j]
        return img

    def permuteCol(self,img,half_index):
        # half_index=self.N//2
        for j in range(0,half_index,2):
            for i in range(half_index*2):
                img[i][j],img[i][half_index+j]=img[i][half_index+j],img[i][j]
        return img



    def decrptionStuffing(self,img,half_len):
        c=0
        for i in range(3):
            img=np.rot90(img,-1)
            img=self.diagnolStuffing(img,half_len*2)
        for i in range(1,2):
            # img=np.rot90(img,-1)
            # img=self.diagnolStuffing(img)
            img=self.permuteCol(img,half_len)
            # img=np.rot90(img,-2)
            img=self.permuteRow(img,half_len)
            
            
            print(self.label+': Decription Stuffing iteration...')

        return img




    def showImage(self,img,lab):
        try:
            cv2.imwrite(lab+'.png',img)
            cv2.imshow(lab,cv2.imread(lab+'.png'))
            
        except Exception as e:
            print(f'Found error in {lab} \t',e)



    def dec(self,bitSequence):
        return int(bitSequence,2)



    def getHenonMap(self,m,x,y):

        #initial Parameter
        a,b=1.408,0.3
        # x = 0.4
        # y = 0.4
        count=1

        x_arr=[x]
        y_arr=[y]

        sequenceSize = m * m * 8  
        bitSequence = ''   
        byteArray = []  
        res_mat = []   # To store the result

        for i in range(sequenceSize):
            # Henon Map formula
            curr_x = 1 - a * x**2+y
            curr_y = b * x

            x_arr.append(curr_x)
            y_arr.append(curr_y)

            # For next iteration
            x = curr_x
            y = curr_y

            # Each Value is converted into 0 or 1 based on a threshold level
            bit=('0' if curr_x <= 0.38 else '1')
            bitSequence+=bit

            if i % 8 == 7:
                # for each 8 iteration a decimal value is generated
                decimal = self.dec(bitSequence)

                byteArray.append(decimal)
                # print(bitSequence,decimal)

                # Clear to store new Value
                bitSequence = ''


            #ByteArray is inserted into res_mat
            byteArraySize = m*8
            if i % byteArraySize == byteArraySize-1:
                # print(byteArray)
                print(self.label+': henon map')
                res_mat.append(byteArray)
                # print(len(byteArray),byteArray)
                byteArray = []
                count+=1
        
        plt.rcParams['agg.path.chunksize'] = 10000
        plt.scatter(x_arr[:6000],y_arr[:6000])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f'Henon Map at a=1.4 b=0.3')
        plt.show()

        return res_mat



    def rand_list(self,p, r):
        l = []
        for i in range(p):
            x = np.random.uniform(-r, r)
            y = np.random.uniform(-r, r)
            l.append([x, y])

        return l




    def getIkedaMap(self,points, num_iter, u, r,img):
        #generate p amount of points [x,y], apply the function for n iterations
        #with u as the parameter value for the ikeda map
        xinit = []
        yinit = []


        l = self.rand_list(points, r)
        # l=np.concatenate((img,img),axis=0).reshape(points,2)
        # l=img[:,:2]
        # print('Shape of concatenated matix',l.shape)


        #coping genearted list with new object
        initial_list = l[:]
        images = []
        iter_count=1

        for k in range(len(initial_list)):
            xinit.append(initial_list[k][0])
            yinit.append(initial_list[k][1])
        


        plt.scatter(xinit, yinit)
    

        plt.savefig("frame" + str(0) + ".png")
        images.append(imageio.imread("frame" + str(0) + ".png"))
        plt.clf()

        res_mat=[]
        for i in range(num_iter):
            x_arr = []
            y_arr = []
            for coord in range(len(l)):
                bit=''
                for j in range(8):
                    tn=0.4 - 6 / (1 + l[coord][0]**2 + l[coord][1]**2)
                    xn = 1 + u * (l[coord][0] * math.cos(tn) - l[coord][1] * math.sin(tn))
                    yn = u * (l[coord][0] * math.sin(tn) + l[coord][1] * math.cos(tn))
                    bit+=('1' if int(xn)%2==0 else '0')
                    # x_arr.append(xn)
                    # y_arr.append(yn)
                    # plt.scatter(x_arr,y_arr)
        
                    # plt.savefig("frame" + str(iter_count) + ".png")
                    # plt.clf()
                    # images.append(imageio.imread("frame" + str(iter_count) + ".png"))
                    iter_count+=1
                l[coord] = [xn,yn]
                
                if i==0:
                    res_mat.append(self.dec(bit))
                else:
                    res_mat[coord]=abs(res_mat[coord]-self.dec(bit))
            print(self.label+': Ikeda iteration',i+1)
        imageio.mimsave('final.gif', images)
        plt.subplot(121)
        plt.scatter(xinit, yinit)
        plt.title('initial values of ikedaMap')
        plt.subplot(122)
        plt.scatter(x_arr, y_arr)

        plt.title('ikedaMap after ' + str(num_iter) + ' iterations')
        plt.show()
        print('IkedaMap',len(x_arr),len(y_arr),len(np.array(x_arr)+np.array(y_arr))) 


        res_mat=np.array(res_mat).reshape(64,64)
        print('Shape before concatenate',res_mat.shape)
        
        res_mat=self.cvtTo256(res_mat,sum(res_mat.shape))

        print('Shape after concatenate',res_mat.shape)
        # for i in res_mat:
        #     print(i)
        print('*'*30)
        
        return res_mat

    def cvtTo256(self,img,size):
        # img=np.array(img).reshape(size,size)
        print(self.label+': Converting To original size...')
        img=np.concatenate((img,img),axis=1)
        img=np.concatenate((img,img),axis=0)
        if size<256:
            img=self.cvtTo256(img,sum(img.shape))
        return img

    def rowStuffing(self,img,k):
        print(self.label+': rowStuffing...')
        img=np.roll(img,k,axis=0)
        # self.showImage(img, 'Row Stuffing')
        return img


    def columnStuffing(self,img,k):
        print(self.label+': colStuffing...')
        img=np.roll(img,k,axis=1)
        # self.showImage(img, 'Col Stuffing')
        return img

    def diagnolStuffing(self,img,l):
        print(self.label+': diagonalStuffing...')
        # l=self.N
        i,j=l-1,0
        while j<l:
            img[j][j],img[i][j]=img[i][j],img[j][j]
            i-=1
            j+=1
        return img
    
    def generateSBox(self):
        mat = np.arange(self.N*self.N).reshape(self.N,self.N)
        mat=self.Stuffing(mat,self.N//2)
        return mat

    def histogramAnalysis(self,img1,img2,title1,title2):
        print(self.label+': histogram')
        plt.subplot(121)
        plt.hist(img1.ravel(),256,[0,256]) 
        plt.title(title1)
        plt.xlabel('pixels')
        plt.ylabel('intensity')
        plt.ylim([0,1200])
        plt.subplot(122)
        plt.hist(img2.ravel(),256) 
        plt.title(title2)
        plt.xlabel('pixels')
        plt.ylabel('intensity') 
        plt.ylim([0,1200])  
        plt.show()



def calEntropy(img):
    count={}
    res_entro=0
    for i in range(256):
        count[i]=0

    for i in img:
        for j in i:
            for k in j:
                count[k]+=1
    
    for i in range(256):
        probality=count[i]/(256*256*3)
        try:
            res_entro+=probality*(math.log2(1/probality))
        except:
            continue

    return res_entro


def avg(mat):
    count=0
    total=0
    for i in mat:
        for j in i:
            count+=1
            total+=j
    return total/(len(mat)*len(mat[0]))



img=cv2.imread('peppers.tif')
img=cv2.resize(img,(256,256))
cv2.imshow('Cloud',img)

entropy_original=calEntropy(img)
# print(img)
# print(img.shape)
# m=0
# for i in img:
#     m=max(m,max(i))
# print('max',m)

r,g,b=cv2.split(img)

# cv2.imshow('Cloud_red',r)

chaotic_red=ChaoticProcess(r,'red')


chaotic_red.getIkedaMap(64*64, 50, 0.92, 10, r)

# red_encrypted=chaotic_red.encrypt()
# red_decryted=chaotic_red.decrypt()

# cv2.imshow('Cloud_green',g)
# cv2.imwrite('green.png',g)
# chaotic_green=ChaoticProcess(g,'green')
# green_encrypted=chaotic_green.encrypt()
# green_decrypted=chaotic_green.decrypt()


# cv2.imshow('Cloud_blue',b)
# cv2.imwrite('blue.png',b)
# chaotic_blue=ChaoticProcess(b,'blue')
# blue_encrypted=chaotic_blue.encrypt()
# blue_decrypted=chaotic_blue.decrypt()

# encrypted_rgb=np.dstack((red_encrypted,green_encrypted,blue_encrypted)) 
# chaotic_blue.showImage(encrypted_rgb,'final_Encrypted')

# rgb = np.dstack((red_decryted,green_decrypted,blue_decrypted)) 
# chaotic_blue.showImage(rgb,'DecryptedColor')


# entropy_encrypted=calEntropy(encrypted_rgb)
# entropy_decrypted=calEntropy(rgb)


# print('Entropy of original image is ',entropy_original)
# print('Entropy of encrypted image is ',entropy_encrypted)
# print('Entropy of decrypted image is ',entropy_decrypted)

cv2.waitKey(0)
cv2.destroyAllWindows()