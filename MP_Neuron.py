# NOT GATE
Theta=0
w=-1
sum=0
a=[[0,1],[1,0]]
def threshold(Theta, sum):
  if(Theta<=sum):
    return 1
  else:
    return 0

# OR GATE
Theta=0
w=-1
sum=0
a=[[0,0,0],[0,1,0],[1,0,0],[1,1,1]]
for i in range(len(a)):
  sum=0
  for j in range(len(a[i])-1):
    sum+=a[i][j]*w
  print("OR({},{})={}".format(a[i][0],a[i][1], threshold(1,sum)))

# AND GATE
Theta=0
sum=0
w=1
a=[[0,0,0],[0,1,0],[0,0,1],[1,1,1]]
for i in range(len(a)):
  sum=0
  for j in range(len(a[i])-1):
    sum+=a[i][j]*w
  print("AND({},{})={}".format(a[i][0],a[i][1],threshold(0,sum)))