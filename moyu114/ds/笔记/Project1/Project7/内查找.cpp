#include<iostream>
using namespace std;
typedef int KeyType;
typedef char InfoType;
typedef struct {
	KeyType key;
	InfoType data;
}RecType;
void InsertSort(RecType R[], int n) {
	int i, j;
	RecType tmp;
	for (int i = 1;i < n;i++) {
		if (R[i].key < R[i - 1].key) {//如果反序
			tmp = R[i];
			j = i - 1;    //
			do {           //从尾部寻找插入位置
				R[j + 1] = R[j];
				j--;
			} while (j >= 0 && R[j].key > tmp.key);
			R[j + 1] = tmp;
		}
	}
}
void BinInsertSort(RecType R[], int n) {
	int i, j, low, high, mid;
	RecType tmp;
	for (int i = 1;i < n;i++) {
		if (R[i].key < R[i - 1].key) {//反序时
			tmp = R[i];
			low = 0;high = i - 1;
			while (low <= high) {//在R[low..high]中查找插入的位置
				mid = (low + high) / 2;//取中间位置
				if (tmp.key < R[mid].key)
					high = mid - 1;//插入点在左半区
				else
					low = mid + 1;//插入点在右半区
			}                    //找位置high
			for (j = i - 1;j > high + 1;j--)//记录后移
				R[j + 1] = R[j];
			R[high + 1] = tmp;//插入tmp
		}
	}
}
void ShellSort(RecType R[], int n) {
	int i, j, d;
	RecType tmp;
	d = n / 2;
	while (d > 0) {
		for (i = d;i < n;i++) {//对相隔d位置的元素组直接插入排序
			tmp = R[i];
			j = i - d;
			while (j >= 0 && tmp.key < R[j].key) {
				R[j + d] = R[j];
				j = j - d;
			}
			R[j + d] = tmp;
		}
		d = d / 2; //减小增量
	}
}
void BubbleSort(RecType R[],int n)
{
	int i,j;  RecType temp;
	for (i = 0;i < n - 1;i++)
	{
		for (j = n - 1;j > i;j--)  	//从尾部比较，找本趟最小关键字的记录
			if (R[j].key < R[j - 1].key)
			{
				temp = R[j];    	//R[j]R[j-1]
				R[j] = R[j - 1];
				R[j - 1] = temp;
			}
	}
}
void QuickSort(RecType R[],int s,int t){//对R[s]至R[t]的元素进行快速排序
	int i = s, j = t;
	RecType tmp;
	if (s < t) {//区间内至少存在2个元素的情况
		tmp = R[s];  //用区间的第1个记录作为基准
		while (i != j) {  //两端交替向中间扫描，直至i=j为止
			while (j > i&& R[j].key >= tmp.key)j--;
			R[i] = R[j];
			while (i < j && R[j].key <= tmp.key)i++;
			R[j] = R[i];
		}R[i] = tmp;
		QuickSort(R, s, i - 1);//对左区间进行排序
		QuickSort(R, i + 1, t);//对右区间进行排序
	}
	//递归出口：不需要任何操作

}
int Min(int a[], int n, int i) {
	int k = i, j;
	for (int j = i+1;j < n;j++) {
		if (a[j] < a[k])k = j;
		return a[k];
	}
}
void SelectSort(RecType R[], int n) {
	int i, j, k;RecType tmp;
	for (i = 0;i < n - 1;i++) {
		k = i;
		for(j=i+1;j<n;j++)
			if (R[j].key < R[k].key) { 
				k = j;	
			}
		if (k != i) {
			tmp = R[i];  R[i] = R[k];  R[k] = tmp;
		}
	}
}
void sift(RecType R[], int low, int high) {//调整堆的算法
	int i = low, j = 2*i;//R[j]是R[i]的左孩子
	RecType tmp = R[i];
	while (j <= high) {
		if (j < high && R[j].key < R[j + 1].key)j++;
		if (tmp.key < R[j].key) {//双亲小
			R[i] = R[j];     //将R[j]调整到双亲结点位置上
			i = j;          //修改i和j值，以便继续向下筛选
			j = 2 * i;
		}
		else
			break;//双亲大：不再调整
	}R[i] = tmp;
}
void HeapSort(RecType R[],int n)
{
	int i;  RecType tmp;
	for (i = n / 2;i >= 1;i--) 	//循环建立初始堆
		sift(R,i,n);
	for (i = n; i >= 2; i--)	//进行n-1次循环，完成堆排序
	{
		tmp = R[1];       	//R[1] R[i]
		R[1] = R[i];  R[i] = tmp;
		sift(R,1,i - 1);   	//筛选R[1]结点，得到i-1个结点的堆
	}
}
void Merge(RecType R[], int low, int mid, int high)//将两个表直接拼接在一起，再放回R中
{
	RecType* R1;
	int i = low, j = mid + 1, k = 0;
	//k是R1的下标，i、j分别为第1、2段的下标
	R1 = (RecType*)malloc((high - low + 1) * sizeof(RecType));
	while (i <= mid && j <= high)
		if (R[i].key <= R[j].key)  	//将第1段中的记录放入R1中
		{
			R1[k] = R[i];  i++;k++;
		}
		else   //将第2段中的记录放入R1中
		{
			R1[k] = R[j];  j++;k++;
		}
	while (i <= mid)         //将第1段余下部分复制到R1
	{
		R1[k] = R[i];  i++;k++;
	}
	while (j <= high)        //将第2段余下部分复制到R1
	{
		R1[k] = R[j];  j++;k++;
	}
	for (k = 0,i = low;i <= high;k++,i++) 	//将R1复制回R中
		R[i] = R1[k];
	free(R1);
}
void MergePass(RecType R[],int length,int n)//一趟二路归并（段长度为length ）
{
	int i;
	for (i = 0;i + 2 * length - 1 < n;i = i + 2 * length)   //归并length长的两相邻子表 	
		Merge(R,i,i + length - 1,i + 2 * length - 1);
	if (i + length - 1 < n) 		      //余下两个子表，后者长度小于length
		Merge(R,i,i + length - 1,n - 1);  	//归并这两个子表
}
void MergeSort(RecType R[], int n)
{
	int length;
	for (length = 1;length < n;length = 2 * length)//每次归并长度逐渐增加
		MergePass(R, length, n);
}
#define MAXE 20		//线性表中最多元素个数
#define MAXR 10		//基数的最大取值
#define MAXD 8		//关键字位数的最大取值
typedef struct node
{
	char data[MAXD];		//记录的关键字定义的字符串
	struct node* next;
}  RecType1;//单链表中每个结点的类型
void RadixSort(RecType1*& p,int r,int d)
//p为待排序序列链表指针，r为基数，d为关键字位数
{
	RecType1* head[MAXR],* tail[MAXR],* t;  //定义各链队的首尾指针
	int i, j, k;
	for (i = 0;i < d;i--)       		//从低位到高位做d趟排序
	{
		for (j = 0;j < r;j++)       	//初始化各链队首、尾指针
			head[j] = tail[j] = NULL;
		//分配
		while (p != NULL)         	//对于原链表中每个结点循环
		{
			k = p->data[i] - '0'; 	//找第k个链队
			if (head[k] == NULL) 	//进行分配，即采用尾插法建立单链表
			{
				head[k] = p;  tail[k] = p;
			}
			else
			{
				tail[k]->next = p;  tail[k] = p;
			}
			p = p->next;       	 //取下一个待排序的结点
		}
		p = NULL;
		//收集
		for (j = 0;j < r;j++)     	//对于每一个链队循环进行收集
			if (head[j] != NULL)
			{
				if (p == NULL)
				{
					p = head[j];
					t = tail[j];
				}
				else
				{
					t->next = head[j];
					t = tail[j];
				}
			}
		t->next = NULL;  	//最后一个结点的next域置NULL
	}
}
int main() {
	RecType R[10];
	int a[] = { 6,8,7,9,0,1,3,2,4,5 };
	for (int i = 0;i < 10;i++) {
		R[i].key = a[i];
	}
	QuickSort(R, 0, 9);
}