#include "base.h"
#include "arena.h"
#include "prng.h"
#include "arena.c"
#include "prng.c"

typedef struct 
{
    u32 rows, cols;
    f32* data;
} matrix;

matrix* mat_create(mem_arena* arena, u32 rows, u32 cols);

int main(void)
{
    mem_arena* perm_arena = arena_create(GiB(1), MiB(1));

    arena_destroy(perm_arena);

    return 0;
}