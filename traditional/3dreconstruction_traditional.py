# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 19:06:50 2022

@author: mitran
"""
import cv2 
import numpy as np
from numpy.linalg import svd
import open3d as o3d
import matplotlib.pyplot as plt





def skew(a):

    return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,_ = img1.shape
    
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),10,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),10,color,-1)
    return img1,img2

class Reconstruct3d():
    def __init__(self,feature_detector='sift'):
        
        # feature detector and extractor
        if(feature_detector=='sift'):
            self.feature_detector=cv2.SIFT_create()
        elif(feature_detector=='sift'):
            self.feature_detector=cv2.SIFT_create()
            
        #features matching
            
        ip = dict(algorithm = 0, trees = 5)#algorithm FLANN_INDEX_KDTREE
        sp = dict(checks=100)  
        self.matcher= cv2.FlannBasedMatcher(ip,sp)
        
        
        
        
        
    
        
            
    def Fundamental_matrix(self,pts1,pts2):
        
        #epipolar constraint  x' F x = 0
        
        F, _ = cv2.findFundamentalMat(pts1,pts2)
        
        return F
    
    
    def Projection_matrix(self,F):
        
        
        """The (right) null space of A is the columns of V corresponding to singular values equal to zero.
        The left null space of A is the rows of U corresponding to singular values equal to zero"""
        
        # F . e = 0,  e is the null-vector of F
        # FT . e' = 0,  e is the null-vector of F
        
        # 1 column has 1 vector
        
        U,S,VT=svd(F)      
        epipole_1= VT[-1]
        epipole_1=epipole_1/epipole_1[2]
        
        U,S,VT=svd(F.T)
        epipole_2= VT[-1]       
        epipole_2=epipole_2/epipole_2[2] # e'
        
        
        #  P1 = [I | 0] P2 = [[e’]xF | e’]
        # shape ->(3,4)
        
        P1 = np.hstack((np.eye(3),np.zeros((3,1))))
        
        skew_e2=skew(epipole_2)
        
        epipole_2=epipole_2.reshape(-1, 1)
        
        
        
        P2 = np.hstack((np.dot(skew_e2 , F) ,epipole_2))
        print(P2)
        
        return P1,P2
    
    def Triangulation(self,pts1,pts2,P1,P2):
        
        # x= PX
        # x crossproduct PX =0 -> Cross product of two vectors of same direction is zero 
        
        N=len(pts1)
        
        coordinates_3D=[]
        
        
        for i in range(N):
            
            x1=pts1[i]
            x2=pts2[i]
            
            x1, y1 = pts1[i, 0], pts1[i, 1]
            x2, y2 = pts2[i, 0], pts2[i, 1]
            
            T=np.zeros((4,4))
            
            T[0,:] = x1 * P1[2,:]- P1[0,:] 
            T[1,:] = P1[1,:] - y1 * P1[2,:] 
            
            T[2,:] = x2 * P2[2,:]- P2[0,:] 
            T[3,:] = P2[1,:] - y2 * P2[2,:] 
            
            
            U,S,VT = svd(T)
            X = VT[-1,:]
            X = X/X[-1]
            
            
           
            #assert(1==2)
            
      
            
    
            
            coordinates_3D.append(X)
            
        return np.array(coordinates_3D)
    
    
    def cvt_homo(self,x):
        
        return np.hstack([x,np.ones((len(x),1))])
        
    def get_depth(self,P1,P2,coordinates):
        
        c1 = np.array([0,0,0,1])
        c2 = np.append(-np.dot(np.linalg.inv(P2[:3, :3]), P2[:, 3]), [1])
        
        diff_c1 = coordinates - c1
        diff_c2 = coordinates - c2
        
        princ_axis_1 = np.append(P1[2,:3], [0])
        princ_axis_2 = np.append(P2[2,:3], [0])
        
        D1 = np.dot(diff_c1, princ_axis_1)
        D2 = np.dot(diff_c2, princ_axis_2)
    
        D1 = np.linalg.det(P1[:3,:3]) * D1
        D2 = np.linalg.det(P1[:3,:3]) * D2
        
        return D1,D2
        
        
    def Draw_lines(self,img1,img2,pts1,pts2,F):
        
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
        lines1 = lines1.reshape(-1,3)
        img4,_ = drawlines(img1,img2,lines1,pts1,pts2)
        
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
        lines2 = lines2.reshape(-1,3)
        img3,_ = drawlines(img2,img1,lines2,pts2,pts1)
        
        img3=cv2.resize(img3,(1000,1000))
        img4=cv2.resize(img4,(1000,1000))
        
        
        cv2.imwrite("epipolarlines.jpg",np.hstack([img4,np.zeros((1000,100,3)),img3]))
        
    def get_allpoints(self,shape,pts1,pts2):
        
        
        """
        estimate homography with given points
        and use the homography to find correspondance for all points in the image
        """
        
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC,5.0)
        H=H.T
        
        
        
        h,w=shape[:2]
        
        img1_points=pts1#np.array([[[r,c] for c in range(h)] for r in range(w)]).reshape(-1,2)
        img1_points=self.cvt_homo(img1_points)
        
        img2_points=np.dot(img1_points,H)
        img2_points/=img2_points[:,2].reshape(-1,1)
        
       
        img1_points=img1_points[:,:2]
        img2_points=img2_points[:,:2]
        
        return img1_points,img2_points
        
    def reconstruct(self,image_1,image_2):
        
        img1=image_1
        img2=image_2
        
        kp1,feat_1=self.feature_detector.detectAndCompute(img1,None)
        kp2,feat_2=self.feature_detector.detectAndCompute(img2,None)
        
        matches = self.matcher.knnMatch(feat_1,feat_2,k=2)

        pts1=[]
        pts2=[]
        match_filtered=[]
        
        additional_pts1=[]
        additional_pts2=[]
        
        # Nearest Neighbour distance ratio ([A0-A1]/[A0-A2])
        for i,(m1,m2) in enumerate(matches):
            
            if m1.distance < 0.75 *m2.distance:
                additional_pts1.append(kp1[m1.queryIdx].pt)
                additional_pts2.append(kp2[m1.trainIdx].pt)
            
            if m1.distance < 0.6 *m2.distance:
                
                #queryindex -> m1 match's first image keypoint  index
                #trainindex -> m1 match's second image keypoint index
                
                pts1.append(kp1[m1.queryIdx].pt)
                pts2.append(kp2[m1.trainIdx].pt)
                match_filtered.append(m1)
        
                
        
        
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        
        pts1,pts2=self.get_allpoints(image_1.shape,pts1,pts2)   
        
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        print("no of points choosed  :",len(pts1))
        
        Fundamental_matrix=self.Fundamental_matrix(pts1, pts2)        
        print("Fundamental Matrix estimated")
        
        #self.Draw_lines(img1.copy(),img2.copy(),pts1,pts2,Fundamental_matrix) 
        
        P1,P2=self.Projection_matrix(Fundamental_matrix)
        print("Camera matrices are Found")
        
        additional_pts1 = np.int32(additional_pts1)
        additional_pts2 = np.int32(additional_pts2)
        
        additional_pts1,additional_pts2=self.get_allpoints(image_1.shape,additional_pts1,additional_pts2)   
        
        additional_pts1 = np.int32(additional_pts1)
        additional_pts2 = np.int32(additional_pts2)
        
        
        coordinates_3D=self.Triangulation(additional_pts1, additional_pts2, P1, P2)  
        
        D1,D2=self.get_depth(P1,P2,coordinates_3D)
       
        
        
        return D1,coordinates_3D,additional_pts1
        
        
        
        
                
        
        
        
                
        
                
            
m=Reconstruct3d()
pth="./Traditional/"
img1=cv2.imread(pth+"image_1.jpeg")    
img2=cv2.imread(pth+"image_2.jpeg")  


D1,D2,kps=m.reconstruct(img1, img2) 
f=24/1000
D1=(np.abs(D1))*(1/f)
for k,i in enumerate(kps):
    img1=cv2.circle(img1,tuple(i),5,(0,255,255),-1)
    

    font = cv2.FONT_HERSHEY_SIMPLEX
  
    org = (i[0]+7, i[1]+7)
  

   
    color = (0, 0, 255)

   
    img1 = cv2.putText(img1, str(int(D1[k])), org, font, 0.5, color, 1, cv2.LINE_AA)
cv2.imwrite("check.jpg",img1)



































     
        
        
        
        
        
    