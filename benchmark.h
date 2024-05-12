#include <iostream>
#include <csignal>
#include <ucontext.h>

// Signal handler to adjust the program counter
static void cca_sighandler(int signo, siginfo_t *si, void *data) {
    ucontext_t *uc = reinterpret_cast<ucontext_t *>(data);
    uc->uc_mcontext.pc += 4;  // Adjust program counter by size of one instruction
}

#define STR(s) #s
#define CCA_MARKER(marker) __asm__ volatile("MOV XZR, " STR(marker))

// Tracing halt and resume using specific opcodes (if supported by architecture)
#define CCA_TRACE_START __asm__ volatile("HLT 0x1337");
#define CCA_TRACE_STOP __asm__ volatile("HLT 0x1337");
 
// Initializes signal handling for illegal instruction used as a marker
#define CCA_BENCHMARK_INIT { \
    struct sigaction sa = {0}; \
    sa.sa_flags = SA_SIGINFO; \
    sa.sa_sigaction = cca_sighandler; \
    sigemptyset(&sa.sa_mask); \
    sigaction(SIGILL, &sa, NULL); \
}

#define CCA_MARKER_START CCA_MARKER(0x1000)
#define CCA_MARKER_END CCA_MARKER(0x1001)

#define CCA_MARKER_INFERENCE_INITIALISATION_START CCA_MARKER(0x2000)
#define CCA_MARKER_INFERENCE_INITIALISATION_END CCA_MARKER(0x2001)

#define CCA_MARKER_READ_INPUT_START CCA_MARKER(0x2100)
#define CCA_MARKER_READ_INPUT_STOP CCA_MARKER(0x2101)

#define CCA_MARKER_NEW_INFERENCE_START CCA_MARKER(0x2200)
#define CCA_MARKER_NEW_INFERENCE_STOP CCA_MARKER(0x2201)

#define CCA_MARKER_WRITE_OUTPUT_START CCA_MARKER(0x2300)
#define CCA_MARKER_WRITE_OUTPUT_STOP CCA_MARKER(0x2301)

#define CCA_MARKER_UPDATE_STATE_START CCA_MARKER(0x2400)
#define CCA_MARKER_UPDATE_STATE_STOP CCA_MARKER(0x2401)


/*
void cca_event_marker(unsigned int marker_num, int flag){
     if (flag == 1){
	CCA_TRACE_START;
        CCA_MARKER(marker_num);
	CCA_TRACE_END;
     }

}
*/
