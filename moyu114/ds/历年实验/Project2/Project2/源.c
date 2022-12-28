#include<stdio.h>
#include<string.h>
int mergeBar(char str[],char strn[]) {
	
	int p = 0, cnt = 0;
	int flag = 0;
	if (str[0] == '\\' || str[0] == '/') {
		flag = 1;
	}
	while (str[p] != '\0') {
		
		if (str[p] != '\\' && str[p] != '/' && str[p] != '.') {
			strn[cnt] = str[p];
			cnt++;
			p++;
		}
			while (str[p] == '\\' || str[p] == '/'|| str[p]=='.') {
				if (str[p] == '.' && str[p + 1] != '.') {
					p++;
					continue;
				}
				else if(str[p] == '.' && str[p + 1] == '.'){
					strn[cnt] = '.';
					cnt++;
					p++;
					continue;
				}
				else if ((str[p] == '\\' || str[p] == '/') && (str[p-1] != '\\' && str[p-1] != '/'&& str[p - 1] != '.')) {
					strn[cnt] = '/';
					cnt++;
					p++;
					continue;
				}
				p++;
			}
		}
		
	
	strn[cnt] = '\0';
	return flag;
}
void deleteHashtag(char* str) {
	int j = 0;
	for (int i = 0; i < strlen(str); i++) {
		if (str[i] != '#')
			str[j++] = str[i];
	}
	str[j] = '\0';
}
int deleteDoublePoint(char* str) {
	int i = 0;
	while (str[i]!='\0') {
		if (str[i] == '.') {
			if (i - 2 < 0) {
				return 0;
			}
			str[i] = str[i - 1] = '#';
			if (str[i + 1] == '/')
				str[i + 1] = '#';
			i--;
			while (i >= 0 && str[i] != '/')
				str[i--] = '#';
			deleteHashtag(str);

		}
		i++;
	}
	return 1;
}
void Print(char str[]) {

	int p = 0;
	int len = strlen(str);
	if (str[len - 1] == '/') {
		str[len - 1] = '\0';
	}
	while (str[p]=='.'|| str[p] == '/'||str[p] == '\\') {
		p++;
	}
	while (str[p]!='\0') {
		printf("%c", str[p]);
		p++;
	}
	printf("\n");
}
char* getOnce(char* str) {
	char c;
	int i = 0;
	while ((c = getchar()) != '\n' && c != EOF)
		str[i++] = c;
	str[i] = '\0';
	if (c == '\n')
		return str;
	else
		return NULL;
}

int main() {
	char str1[10000];
	char str2[10000];

	int flag1, flag2;
	char* sign;
	do {
		sign = getOnce(str1);
		flag1 = mergeBar(str1, str2);
		flag2 = deleteDoublePoint(str2);
		if (!flag2) {
			printf("Value Error\n");
		}
		else if (flag1) {
			printf("/");
			Print(str2);
		}
		else {
			Print(str2);
		}
	} while (sign);
}