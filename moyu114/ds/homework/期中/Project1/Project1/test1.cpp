#include<iostream>
#include<malloc.h>
#include<math.h>
using namespace std;
void arrange(int arr[], int idx, int N, int& tree_count) {
	int i=0,lchild,rchild;
	if (N == idx) {//��������
		tree_count++;
		cout << tree_count << ":";
		for (int i = 1;i <= N;i++) {
			cout << arr[i] << " ";
		}
		cout << endl;
		return;
	}
	for (i = 1;i <= idx;i++) {//��1~idx�����н��ĺ��ӽ���ҳ�
		lchild = arr[i] * 2;
		rchild = arr[i] * 2 + 1;
		if (lchild > arr[idx]) {//lchild��arr[idx]�ڴ�ŵĽ��Ž��бȽ�
			arr[idx + 1] = lchild;
			arrange(arr, idx + 1, N, tree_count);
			arr[idx + 1] = 0; //������Ҫ��ѡ����Ľ�����㣬���ٿ���
		}
		if (rchild > arr[idx]) {//rchild��arr[idx]�ڴ�ŵĽ��Ž��бȽ�
			arr[idx + 1] = rchild;
			arrange(arr, idx + 1, N, tree_count);
			arr[idx + 1] = 0;
		}
	}	
}

int Catalan(int n) {
	int temp=1 ,result;
	for (int i = 1; i <= n; i++) {
		temp *= (2 * n + 1 - i);
		temp /= i;
	}
	result=temp / (1 + n);
	return result;
}
void arrangeCa(int arr[], int idx, int N, int& tree_count) {
	int i = 0, lchild, rchild;
	if (N == idx) {//��������
		tree_count++;
		return;
	}
	for (i = 1;i <= idx;i++) {//��1~idx�����н��ĺ��ӽ���ҳ�
		lchild = arr[i] * 2;
		rchild = arr[i] * 2 + 1;
		if (lchild > arr[idx]) {//lchild��arr[idx]�ڴ�ŵĽ��Ž��бȽ�
			arr[idx + 1] = lchild;
			arrangeCa(arr, idx + 1, N, tree_count);
			arr[idx + 1] = 0; //������Ҫ��ѡ����Ľ�����㣬���ٿ���
		}
		if (rchild > arr[idx]) {//rchild��arr[idx]�ڴ�ŵĽ��Ž��бȽ�
			arr[idx + 1] = rchild;
			arrangeCa(arr, idx + 1, N, tree_count);
			arr[idx + 1] = 0;
		}
	}
}
void arrangeNew(int arr[], int idx, int N, int& tree_count, int& height) {
	int i = 0, lchild, rchild;
	if (N == idx) {//��������
		tree_count++;
		int LastNo = 2;
		while (LastNo <= arr[idx])
			LastNo *= 2;
		height += log2(LastNo);
		return;
	}
	for (i = 1;i <= idx;i++) {//��1~idx�����н��ĺ��ӽ���ҳ�
		lchild = arr[i] * 2;
		rchild = arr[i] * 2 + 1;
		if (lchild > arr[idx]) {//lchild��arr[idx]�ڴ�ŵĽ��Ž��бȽ�
			arr[idx + 1] = lchild;
			arrangeNew(arr, idx + 1, N, tree_count,height);
			arr[idx + 1] = 0; //������Ҫ��ѡ����Ľ�����㣬���ٿ���
		}
		if (rchild > arr[idx]) {//rchild��arr[idx]�ڴ�ŵĽ��Ž��бȽ�
			arr[idx + 1] = rchild;
			arrangeNew(arr, idx + 1, N, tree_count,height);
			arr[idx + 1] = 0;
		}
	}
}
int main() {
	int N, tree_count = 0 ;
	int height = 0;
	int arr[100];

	cout << "(1)arrange�����Ĳ���" << endl;
	cout << "N=";
	cin >> N;	
	arr[1] = 1;
	arrange(arr, 1, N, tree_count);
	printf("tree_count is %d when N is %d\n", tree_count, N);

	cout << "(2)��֤M��N֮�����㿨�������Ĺ�ϵ(Nȡ1~10)" << endl;
	for (int i = 1;i <= 10;i++) {
		tree_count = 0;//��ʼ��
		cout << "N=" << i << " ";
	arrangeCa(arr, 1, i, tree_count);
	printf( "Catalan=%d,M=%d",Catalan(i),tree_count);
	(tree_count == Catalan(i)) ? cout << " Match!" << endl : cout << " Not Match" << endl;
	}

	cout << "(3)����arrange��������֤��������ƽ���߶���log2�Ĺ�ϵ(Nȡ1~10)" << endl;
	arrangeNew(arr, 1, N, tree_count,height);
	for (int i = 1;i <= 10;i++) {
		tree_count = 0;
		height = 0;//��ʼ��
		arrangeNew(arr, 1, i, tree_count, height);
		
		cout << "N=" << i<<" ";
		cout <<"total height="<< height<<" ";
		float avgheight = (float)height / tree_count;
		printf("average height=%f,log2N=%f\n", avgheight, log2(i));
	}
	

}
