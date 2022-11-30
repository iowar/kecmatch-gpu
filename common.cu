#include "common.cuh"

void usage(char *arg) {
    fprintf(stderr,
	    "Usage   : %s -f <function_name> -a <function_args> -s "
	    "<function_signature>\n",
	    arg);

    fprintf(stderr,
	    "Example : %s -f withdraw -a \"(uint256,address)\" -s "
	    "0xab0202ba\n",
	    arg);
}

__host__ __device__ size_t _strlen(char s[]) {
    size_t i;

    i = 0;
    while (s[i] != '\0')  // loop over the array, if null is seen, break
	++i;
    return i;  // return length (without null)
}

__device__ int _itoa(int value, char *sp, int radix) {
    char tmp[16];
    char *tp = tmp;
    int i;
    unsigned v;

    int sign = (radix == 10 && value < 0);
    if (sign)
	v = -value;
    else
	v = (unsigned)value;

    while (v || tp == tmp) {
	i = v % radix;
	v /= radix;
	if (i < 10)
	    *tp++ = i + '0';
	else
	    *tp++ = i + 'a' - 10;
    }

    int len = tp - tmp;

    if (sign) {
	*sp++ = '-';
	len++;
    }

    while (tp > tmp) *sp++ = *--tp;

    return len;
}
