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
            if (lchild > arr[idx+1] && lchild > arr[idx]) {//�ж������Ƿ����������Ҳ���֮ǰ�Ľ��ظ�
                idx++;
                arr[idx] = lchild;
                flag = false;
                break;
            }
            else if (rchild > arr[idx+1] && rchild > arr[idx]) {//�ж��Һ����Ƿ����������Ҳ���֮ǰ�Ľ��ظ�
                idx++;
                arr[idx] = rchild;
                flag = false;
                break;
            }
            else if (lchild <= arr[idx] || rchild <= arr[idx])//�ж��Ƿ��֮ǰ�Ľ��ظ������ظ�ֱ�ӽ�����ǰѭ����������һ��ѭ��
                continue;
            else                                           //��ǰλ���ϵ����кϷ�ȡֵ���Ѿ��������
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