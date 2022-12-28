#include<iostream>
#include<malloc.h>
#include<math.h>
using namespace std;
void buildtree(int N, int& tree_count) {
    int arr[100];
    arr[0]=arr[1] = 1;
    int idx = 1;
    bool flag = false;
    while (idx > 0) {
        if (idx == N) {
            tree_count++;
            cout << tree_count << ":";
            for (int i = 1;i <= N;i++) {
                cout << arr[i] << " ";
            }
            cout << endl;
            idx--;
        }
        for (int i = 1;i <= idx;i++) {
            int lchild = arr[i] * 2;
            int rchild = arr[i] * 2 + 1;
            if (lchild > arr[idx+1] && lchild > arr[idx]) {//判断左孩子是否满足条件且不跟之前的解重复
                idx++;
                arr[idx] = lchild;
                flag = false;
                break;
            }
            else if (rchild > arr[idx+1] && rchild > arr[idx]) {//判断右孩子是否满足条件且不跟之前的解重复
                idx++;
                arr[idx] = rchild;
                flag = false;
                break;
            }
            else if (lchild <= arr[idx] || rchild <= arr[idx])//判断是否跟之前的解重复，若重复直接结束当前循环，进入下一次循环
                continue;
            else                                           //当前位置上的所有合法取值都已经试验完毕
                flag = true;
        }
        if (flag) {
            arr[idx+1] = 0;
            idx--;
            
        }
    }
}

int main() {

    int N;
    cout << "N=";
    cin >> N;
   
    int tree_count = 0;
    buildtree( N, tree_count);
    printf("tree_count is %d when N is %d\n", tree_count, N);
}