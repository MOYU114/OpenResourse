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


//static void Main(string[] args)//调用
//{
//	for (int j = 1; j < 10; j++)
//		GetArrayResult(j);
//}
//
//static void GetArrayResult(int n)
//{
//	int[] number = new int[2 * n];//定义一个2*n个元素的整型数组
//	List<int[]> list = new List<int[]>();
//	list.Add(number);//未加入任何数字之前，列表只有一个数组，数组中所有的数都为0
//	for (int i = 1; i <= n; i++)//将1到n的数逐一加到数组中
//	{
//		list = GetIntArray(list, i, 2 * n);
//	}
//	Console.WriteLine("当N的值为" + n + "时，共有" + list.Count + "组解：");
//	foreach(int[] array in list)//打印出所有的解
//	{
//		foreach(int num in array)
//			Console.Write(num);
//		Console.WriteLine();
//	}
//}
//
//static List<int[]> GetIntArray(List<int[]> list, int n, int count)
//{
//	List<int[]> temp = new List<int[]>();//声明一个新的数组列表
//	foreach(int[] array in list)//从数组列表中逐个取出数组，给数组加上数n
//	{
//		for (int i = 0; i < count; i++)
//		{
//			int[] tempArray = array.ToArray();
//			//检查数组中第i位和第(i+n+1)位是否为0
//			if (i + n + 1 < count && tempArray[i] == 0 && tempArray[i + n + 1] == 0)
//			{
//				tempArray[i] = n;//给第i位和第(i+n+1)位赋值
//				tempArray[i + n + 1] = n;
//				temp.Add(tempArray);//将新的数组加到temp数组列表中
//			}
//		}
//	}
//	return temp;//返回新的数组列表
//}