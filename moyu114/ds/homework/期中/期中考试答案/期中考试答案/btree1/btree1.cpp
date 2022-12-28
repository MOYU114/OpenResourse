#include <stdio.h>


void disp(int arr[], int N, int total){
    printf("%d: ",total);
    for(int i=0;i<N-1;i++)
        printf("%d, ",arr[i]);
    printf("%d\n",arr[N-1]);
}

void arrange(int arr[],int idx,int N,int &total){
    int start_val = arr[idx-1]+1;
    int start_pos=0;
    for(int i=idx-1;i>=0;i--){
        if(arr[i]<start_val/2){
            start_pos=i+1;
            break;
        }
        else if(arr[i]==start_val/2){
            start_pos=i;
            break;
        }
    }
    for(int i=start_pos;i<idx;i++){
        for(int j=0;j<2;j++){
            int h = 2*arr[i]+j;
            if(h>=start_val){
                arr[idx]= h;
                if(idx==N-1){
                    total++;
                    disp(arr,N,total);
                }
                else
                    arrange(arr,idx+1,N,total);
            }
        }
    }
}

int main(){
    int *arr;
    int N=3;
    arr = new int[N];
    for(int i=0;i<N;i++)
        arr[i]=0;
    int total=0;
    arr[0]=1;
    arrange(arr,1,N,total);
    printf("tree_count is %d when N is %d\n",total,N);
    delete[] arr;
}
