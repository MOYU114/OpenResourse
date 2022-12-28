#include <stdio.h>

bool nextval(int *arr, int idx){
    int val = arr[idx];
    int nv = 0;  //next value
    for(int i= idx-1;i>=0;i--){
        if(arr[i]*2+1<val)
            break;
        int h = arr[i]*2;
        if(h>=val)
            nv = h;
        else if(h+1>=val)
            nv = h+1;
    }
    if(nv>0){
        arr[idx]=nv;
        return true;
    }
    return false;
}

void disp(int arr[], int N, int total){
    printf("%d: ",total);
    for(int i=0;i<N-1;i++)
        printf("%d, ",arr[i]);
    printf("%d\n",arr[N-1]);
}


void buildtree(int N,int &total){
    int *arr = new int[N];
    for(int i=0;i<N;i++)
        arr[i]=0;
    int idx=1;
    arr[0]=1;
    arr[1]=1;
    while(idx>0){
        if(idx>=N){
            total++;
            disp(arr, N, total);
            idx--;
            continue;
        }
        arr[idx]++;
        if(nextval(arr,idx)){
            idx++;
            if(idx<N)
                arr[idx]=arr[idx-1];
        }
        else
            idx--;
    }
    delete[] arr;
}

int main(){
    int N=5;
    int tree_count=0;
    buildtree(N,tree_count);
    printf("tree_count is %d when N is %d\n",tree_count,N);
}
