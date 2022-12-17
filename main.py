from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage
import matplotlib.pyplot as plt
import numpy as np
import math
from config import Config
import cv2

# 设置
opt = Config()
# 下面四个数组临时存储切点对，其中(sx[i],sy[i])与(tx[i],ty[i])表示同一直线所切的一对切点
sx = []
sy = []
tx = []
ty = []
# 切点对之间的距离
dis = []
# 存储最终的切点集
xx = []
yy = []

# 绘制图像，同时获取图像长宽
def open_img():
    # 打开图像
    img = cv2.imread(opt.path)
    # 绘制图像
    cv2.imshow('original', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 将图片转化为numpy数组
    data = np.asarray(img)
    # print(data)
    # 维度转置
    data = np.transpose(data,[2,0,1])
    # print(data)
    # 获取图像的长宽
    opt.l=len(data[0])
    opt.w=len(data[0][0])
    return data
 
# 获取灰度图像的函数，并将函数重命名为pic_one_channel语句
def open_cvmg():    
    src = cv2.imread(opt.path, cv2.IMREAD_GRAYSCALE) # 以灰度图形式打开图像
    data = np.asarray(src) # 将灰度图片转化为numpy数组
    return data
pic_one_channel = open_cvmg()

# DDA算法扫描,(x0,y0)(x1,y1)为起始点坐标
# 返回是否取到切点的标志，以及切点坐标
def scanline(x0, y0, x1, y1):
    if(x0==x1 and y0==y1):
        return 0,0,0
    dx = x1-x0
    dy = y1-y0
    x = float(x0)
    y = float(y0)    
    if(abs(dx) > abs(dy)):
        eps = abs(dx)
    else:
        eps = abs(dy)    
    xIncre = float(dx)/float(eps)
    yIncre = float(dy)/float(eps)
    
    maxc = 0       #最大的灰度值
    minc = 500     #最小的灰度值
    is_cut = 0     #是否是切点
    #切点坐标
    r = 0
    c = 0
    
    for k in range(0, eps+1):
        #是否出界
        if(int(x) >= opt.l or int(y) >= opt.w):
            continue
        tem = pic_one_channel[int(x)][int(y)]
        if maxc < tem:
            maxc = tem
            r = int(x)
            c = int(y)
        minc = min(minc,tem)
        
        x += xIncre
        y += yIncre
    if(maxc-minc >= opt.thre):
        is_cut = 1
    return is_cut,r,c

# 水平&垂直线扫描
def prescan():
    l = opt.l
    w = opt.w
    # 从上到下
    for i in range(l):
        is_cut = 0
        minc = 500
        maxc = 0
        r = 0
        c = 0
        for j in range(w):
            tem = pic_one_channel[int(i)][int(j)]
            if maxc < tem:
                maxc = tem
                r = int(i)
                c = int(j)
            minc = min(minc,tem)
        if(maxc-minc >= opt.thre):
            is_cut = 1
        if(is_cut):
            sx.append(r)
            sy.append(c)
            break
        
    # 从下到上
    for i in range(l-1,0,-1):
        is_cut = 0
        minc = 255
        maxc = 0
        r = 0
        c = 0
        for j in range(w):
            tem = pic_one_channel[int(i)][int(j)]
            if maxc < tem:
                maxc = tem
                r = int(i)
                c = int(j)
            minc = min(minc,tem)
        if(maxc-minc >= opt.thre):
            is_cut = 1
        if(is_cut):
            tx.append(r)
            ty.append(c)
            break
    
    # 从左到右
    for j in range(w):
        is_cut=0
        minc=255
        maxc=0
        r=0
        c=0
        for i in range(l):
            tem=pic_one_channel[int(i)][int(j)]
            if maxc<tem:
                maxc=tem
                r=int(i)
                c=int(j)
            minc=min(minc,tem)
        if(maxc-minc>=opt.thre):
            is_cut=1
        if(is_cut):
            sx.append(r)
            sy.append(c)
            break    
    
    # 从右到左
    for j in range(w-1,0,-1):
        is_cut=0
        minc=255
        maxc=0
        r=0
        c=0
        for i in range(l):
            tem=pic_one_channel[int(i)][int(j)]
            if maxc<tem:
                maxc=tem
                r=int(i)
                c=int(j)
            minc=min(minc,tem)
        if(maxc-minc>=opt.thre):
            is_cut=1
        if(is_cut):
            tx.append(r)
            ty.append(c)
            break  

#参数方程法绘制圆形，r为半径，(u,v)为圆心
def draw_circle(r,u,v):
    theta = np.arange(0, 2*np.pi, 0.01)
    x = u + r * np.cos(theta)
    y = v + r * np.sin(theta)
    #这里圆是红色的
    plt.plot(x,y,'r')

# 矫正函数l
def ll(omiga, fai):
    x = math.sin(omiga)*math.sqrt(math.cos(fai)**2+(1-math.sin(fai))**2)
    y = math.sin(math.pi-omiga-math.atan((1-math.sin(fai))/abs(math.cos(fai))))
    return x/y

# 插值处理，data是二维数组,表示图片,n,m表示大小
def inter(data, n, m):
    datat = np.copy(data)
    datas = np.copy(data)
    for j in range(m):
        sta=0
        if(data[0][j]==0):
            for i in range(n):
                if(data[i][j]!=0):
                    sta=i
                    break
            for i in range(sta,0,-1):
                datat[i][j]=data[sta][j]
        end=0    
        if(data[n-1][j]==0):
            for i in range(n-1,0,-1):
                if(data[i][j]!=0):
                    end=i
                    break
            for i in range(end,n,1):
                datat[i][j]=data[end][j]
        for i in range(n):
            if(data[i][j]!=0):
                x=data[sta][j]
                y=data[i][j]
                for k in range(sta,i,1):
                    datat[k][j]=x+int((y-x)*(float(k-sta)/float(i-sta))) 
                sta=i
    
    for i in range(n):
        sta=0
        if(data[i][0]==0):
            for j in range(m):
                if(data[i][j]!=0):
                    sta=j
                    break
            for j in range(sta,0,-1):
                datas[i][j]=data[i][sta]
        end=0    
        if(data[i][m-1]==0):
            for j in range(m-1,0,-1):
                if(data[i][j]!=0):
                    end=j
                    break
            for j in range(end,m,1):
                datas[i][j]=data[i][end]
        for j in range(m):
            if(data[i][j]!=0):
                x=data[i][sta]
                y=data[i][j]
                for k in range(sta,j,1):
                    datas[i][k]=x+int((y-x)*(float(k-sta)/float(j-sta))) 
                sta=j
 
    for i in range(n):
        for j in range(m):
            
            data[i][j]=int((datas[i][j]+datat[i][j])/2)
    return data

def main():
    #读取彩色图像
    open_img()
    #先扫描水平&垂直的直线
    prescan()
    
    l = opt.l
    w = opt.w
    th = 0
    
    #直线每次变化的角度
    step=opt.pi/(opt.n*2)
    
    for i in range(opt.n*2):   
        #斜率
        k=math.tan(th)
        #(fr,fc)为扫描过程中，第一个满足切点条件的点
        #(lr,lc)为扫描过程中，最后一个满足切点条件的点
        #这里斜率相同的两条直线并不是分别从两端向圆逼近，而是穿过整个圆。第一个与最后一个满足切点条件的点为切点
        fr = -1
        fc = -1
        lr = -1
        lc = -1
        if(k<10 and k != 0):
            if(k>0):
                #平移直线
                for x0 in range(l,0,-1):
                    if(int(k*(l-x0))<=w):
                        is_cut,r,c=scanline(x0,0,l,int(k*(l-x0)))
                    elif(int(w/k+x0)<=l):
                        is_cut,r,c=scanline(x0,0,int(w/k+x0),w)
                    if(is_cut and fr==-1):
                        fr=r
                        fc=c
                    if(is_cut):
                        lr=r
                        lc=c
                for y0 in range(w):
                    if(int(k*l+y0)<=w):
                        is_cut,r,c=scanline(0,y0,l,int(k*l+y0))
                    elif(int((w-y0)/k)<=l):
                        is_cut,r,c=scanline(0,y0,int((w-y0)/k),w)
                    if(is_cut and fr==-1):
                        fr=r
                        fc=c
                    if(is_cut):
                        lr=r
                        lc=c
            if(k<0):
                for x0 in range(l):
                    if(int(-k*x0)<=w):
                        is_cut,r,c=scanline(x0,0,0,int(-k*x0))
                    elif(int(w/k+x0)<=l):
                        is_cut,r,c=scanline(x0,0,int(w/k+x0),w)
                    if(is_cut and fr==-1):
                        fr=r
                        fc=c
                    if(is_cut):
                        lr=r
                        lc=c
                for y0 in range(w):
                    if(int(y0-k*l<=w)):
                        is_cut,r,c=scanline(0,int(y0-k*l),l,y0)
                    elif(int((w-y0)/k+l)<=l):
                        is_cut,r,c=scanline(int((w-y0)/k+l),w,l,y0)
                    if(is_cut and fr==-1):
                        fr=r
                        fc=c
                    if(is_cut):
                        lr=r
                        lc=c
        if lc!=-1:
            sx.append(fr)
            sy.append(fc)
            tx.append(lr)
            ty.append(lc)
        th+=step
    
    mid_dis=0
    n=len(sx)
    
    #求距离与平均距离
    for i in range(n):
        d=math.sqrt((sx[i]-tx[i])*(sx[i]-tx[i])+(sy[i]-ty[i])*(sy[i]-ty[i]))
        dis.append(d)
    for i in range(n):
        mid_dis+=dis[i]
    mid_dis/=n
    
    #保留距离大于平均距离的点对
    for i in range(n):
        if(dis[i]>=mid_dis):
            xx.append(sx[i])
            xx.append(tx[i])
            yy.append(sy[i])
            yy.append(ty[i])
    
    #圆拟合法
    m=len(xx)
    a=np.ones((m,3))
    b=np.ones((m,1))
    for i in range(m):
        a[i][0]=xx[i]
        a[i][1]=yy[i]
        a[i][2]=1
    for i in range(m):
        b[i][0]=xx[i]*xx[i]+yy[i]*yy[i]
    p=np.matmul(np.linalg.pinv(a),b)
    p1=p[0][0]
    p2=p[1][0]
    p3=p[2][0]
    u0=p1/2
    v0=p2/2
    R=math.sqrt((p1*p1+p2*p2)/4+p3)
    
    #将圆绘制在原图上
    draw_circle(R, v0, u0)

    #截取目标图像，R为圆半径，(u0,v0)为圆心
    img=cv2.imread(opt.path)
    img_valid=img[int(u0-R):int(u0+R+1),int(v0-R):int(v0+R+1)]
    cv2.imwrite(opt.res_path, img_valid)
    
    m,n,k=img_valid.shape[:3]
    result=np.zeros((int(m*3),int(n*3),k))
    l=0
    w=0
    
    for i in range(m):
        for j in range(n):
            
            #经纬变换
            u=j-R
            v=R-i
            r=math.sqrt(u*u+v*v)
            if(r==0):
                fi=0
            elif(u>=0):
                fi=math.asin(v/r)
            else:
                fi = math.pi - math.asin(v/r)
            f = R * 2 / math.pi
            theta = r / f
            x=f * math.sin(theta) * math.cos(fi)
            y=f * math.sin(theta) * math.sin(fi)
            z=f * math.cos(theta)
            
            #sita经度,fai纬度
            rr= math.sqrt(x * x + z * z)
            sita = math.pi / 2 - math.atan( y /rr)
            if(z>=0):
                fai = math.acos(x/rr)
            else:
                fai= math.pi - math.acos(x/rr)
            
            xx=round(f*sita)
            #opt.omiga取Π/2
            h=ll(opt.omiga,0)
            if fai<(math.pi/2):
                yy=round(f*(h-ll(opt.omiga,fai)))
            else:
                yy=round(f*(h+ll(opt.omiga,fai)))
            
            if ((xx < 1) | (yy < 1) | (xx > m) | (yy > n)):
                continue
            l=max(l,xx)
            w=max(w,yy)
            result[xx,yy,0] = img_valid[i, j, 0]
            result[xx,yy,1] = img_valid[i, j, 1]
            result[xx,yy,2] = img_valid[i, j, 2]
    
    result=result[0:l+1,0:w+1]
    
    #插值处理
    result=np.transpose(result,[2,0,1])
    result[0]=inter(result[0],l,w)
    result[1]=inter(result[1],l,w)
    result[2]=inter(result[2],l,w)
    result=np.transpose(result,[1,2,0])
    
    #输出
    Undistortion = np.uint8(result)
    cv2.imwrite(opt.res_path,Undistortion)


if __name__ == '__main__':
    main()

