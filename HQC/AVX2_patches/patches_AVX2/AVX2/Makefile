
CC:=gcc

CFLAGS:=-O3 -std=c99 -funroll-all-loops -flto -mavx -mavx2 -mpclmul -pedantic -Wall -Wextra

RNG_FLAGS:=$(CFLAGS) -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-sign-compare

HASH_SRC:=lib/hash/hash.c
HASH_INCLUDE:=-I lib/hash -lcrypto

SHA3_SRC:=lib/fips202/fips202.c
SHA3_INCLUDE:=-I lib/fips202

RNG_SRC:=lib/rng/rng.c
RNG_INCLUDE:=-I lib/rng

SRC:=src/
INCLUDE:=-I src
LIB:=$(HASH_INCLUDE) $(SHA3_INCLUDE) $(RNG_INCLUDE)

MAIN_HQC:=src/main_hqc.c
MAIN_KAT:=src/main_kat.c
MAIN_BENCH:=src/main_bench.c

HQC_OBJS:=vector.o reed_muller.o reed_solomon.o fft.o gf.o gf2x.o code.o parsing.o hqc.o kem.o
HQC_OBJS_VERBOSE:=vector.o reed_muller.o reed_solomon-verbose.o fft.o gf.o gf2x.o code-verbose.o parsing.o hqc-verbose.o kem-verbose.o
LIB_OBJS:= hash.o rng.o fips202.o

BIN:=bin
BUILD:=bin/build

folders:
	@echo -e "\n### Creating folders\n"
	mkdir -p $(BUILD)

hash.o: folders
	@echo -e "\n### Compiling $@ (wrapper around openssl SHA512 implementation)\n"
	$(CC) $(CFLAGS) -c $(HASH_SRC) $(HASH_INCLUDE) -o $(BUILD)/$@

fips202.o: folders
	@echo -e "\n### Compiling fips202\n"
	$(CC) $(CFLAGS) -c $(SHA3_SRC) $(SHA3_INCLUDE) -o $(BUILD)/$@

rng.o: folders
	@echo -e "\n### Compiling NIST rng.o\n"
	$(CC) $(RNG_FLAGS) -c $(RNG_SRC) $(RNG_INCLUDE) -o $(BUILD)/$@

%.o: $(SRC)/%.c | folders
	@echo -e "\n### Compiling $@\n"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -o $(BUILD)/$@

%-verbose.o: $(SRC)/%.c | folders
	@echo -e "\n### Compiling $@ (verbose mode)\n"
	$(CC) $(CFLAGS) -c $< $(INCLUDE) $(LIB) -D VERBOSE -o $(BUILD)/$@

hqc-128: $(HQC_OBJS) $(LIB_OBJS) | folders
	@echo -e "\n### Compiling hqc-128\n"
	$(CC) $(CFLAGS) $(MAIN_HQC) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $(BIN)/$@

hqc-128-kat: $(HQC_OBJS) $(LIB_OBJS) | folders
	@echo -e "\n### Compiling hqc-128 KAT\n"
	$(CC) $(CFLAGS) $(MAIN_KAT) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $(BIN)/$@

hqc-128-verbose: $(HQC_OBJS_VERBOSE) $(LIB_OBJS) | folders
	@echo -e "\n### Compiling hqc-128 (verbose mode)\n"
	$(CC) $(CFLAGS) $(MAIN_HQC) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -D VERBOSE -o $(BIN)/$@

bench: $(HQC_OBJS) $(LIB_OBJS)
	@echo -e "\n### Compiling HQC Bench\n"
	$(CC) $(CFLAGS) $(MAIN_BENCH) $(addprefix $(BUILD)/, $^) $(INCLUDE) $(LIB) -o $(BIN)/$@

clean:
	rm -f PQCkemKAT_*
	rm -f vgcore.*
	rm -rf ./bin
