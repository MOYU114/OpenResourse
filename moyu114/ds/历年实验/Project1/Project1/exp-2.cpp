#include<iostream>
#include<malloc.h>
using namespace std;

long long fun(int n) {
	long long result=1;
	for (int i = 1;i <= n;i++) {
		result *= i;
	}
	return result;
}
int main() {
	int n;
	cin >> n;
	int c = 2 * n, l = fun(n);
	int** num = (int**)malloc(sizeof(int*) * l);
	for (int i = 0;i < l;i++) {
		num[i] = (int*)malloc(sizeof(int) * c);
		for (int j = 0;j < c;j++)
			num[i][j] = 0;
	}
	int no = 1;
	int cnt = 0;
	int i = 0;
	int pos1 = 0, pos2 = no + 1;
	while (no <= n) {
		i = 0;pos1 = 0, pos2 = no + 1;     
		while (pos2 < c) {
			
			if (num[i][pos1] == 0 && num[i][pos2] == 0) {
				num[i][pos1] = no;
				num[i][pos2] = no;
				cnt++;i++;
				pos1 = 0, pos2 = no + 1;
			}
			else if (num[i][pos1] != 0) {
				if (pos2 >= c) {
					free(num[i]);
					pos1 = 0, pos2 = no + 1;i++;
					cnt--;
				}
				else {
					pos1++;pos2++;
				}


			}
			else if (num[i][pos2] != 0) {
				if (pos2 >=c) {
					free(num[i]);
					pos1 = 0, pos2 = no + 1;i++;
					cnt--;
				}
				else {
					pos1++;pos2++;
				}
			}
		}no++;
	}
	for (int i = 0;i < cnt;i++) {
		for (int j = 0;j < c;j++)
			cout << num[i][j];
		cout << endl;
	}


}


//static void Main(string[] args)//����
//{
//	for (int j = 1; j < 10; j++)
//		GetArrayResult(j);
//}
//
//static void GetArrayResult(int n)
//{
//	int[] number = new int[2 * n];//����һ��2*n��Ԫ�ص���������
//	List<int[]> list = new List<int[]>();
//	list.Add(number);//δ�����κ�����֮ǰ���б�ֻ��һ�����飬���������е�����Ϊ0
//	for (int i = 1; i <= n; i++)//��1��n������һ�ӵ�������
//	{
//		list = GetIntArray(list, i, 2 * n);
//	}
//	Console.WriteLine("��N��ֵΪ" + n + "ʱ������" + list.Count + "��⣺");
//	foreach(int[] array in list)//��ӡ�����еĽ�
//	{
//		foreach(int num in array)
//			Console.Write(num);
//		Console.WriteLine();
//	}
//}
//
//static List<int[]> GetIntArray(List<int[]> list, int n, int count)
//{
//	List<int[]> temp = new List<int[]>();//����һ���µ������б�
//	foreach(int[] array in list)//�������б������ȡ�����飬�����������n
//	{
//		for (int i = 0; i < count; i++)
//		{
//			int[] tempArray = array.ToArray();
//			//��������е�iλ�͵�(i+n+1)λ�Ƿ�Ϊ0
//			if (i + n + 1 < count && tempArray[i] == 0 && tempArray[i + n + 1] == 0)
//			{
//				tempArray[i] = n;//����iλ�͵�(i+n+1)λ��ֵ
//				tempArray[i + n + 1] = n;
//				temp.Add(tempArray);//���µ�����ӵ�temp�����б���
//			}
//		}
//	}
//	return temp;//�����µ������б�
//}