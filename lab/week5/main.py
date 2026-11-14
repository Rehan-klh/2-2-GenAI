
from llm_client import generate_response
from economics import calculate_economics


def print_separator():
    print("=" * 48)


def show_value(label, value):
    print(f"{label:<27}: {value}")


def main():
    print_separator()
    print("       LLM TOKEN ECONOMICS ANALYZER")
    print_separator()

    print("\nSelect Provider:")
    print("1. Google Gemini")
    print("2. Ollama (Local LLM)")

    choice = input("\nEnter choice: ").strip()

    if choice == "1":
        provider = "gemini"
    elif choice == "2":
        provider = "ollama"
    else:
        print("\nInvalid provider selection.")
        return

    prompt = input("\nEnter your prompt:\n").strip()

    if not prompt:
        print("\nPrompt cannot be empty.")
        return

    try:
        result = generate_response(provider, prompt)
        economics = calculate_economics(result)

        # MODEL RESPONSE
        print("\n")
        print_separator()
        print("              MODEL RESPONSE")
        print_separator()
        print(f"\n{result['response']}")

        # TOKEN ANALYSIS
        print("\n")
        print_separator()
        print("              TOKEN ANALYSIS")
        print_separator()

        show_value("Provider", result["provider"])
        show_value("Model", result["model"])
        show_value("Input Tokens", result["input_tokens"])
        show_value("Output Tokens", result["output_tokens"])
        show_value("Total Tokens", result["total_tokens"])
        show_value("Response Latency", f"{result['latency']:.2f} seconds")

        # COST ANALYSIS
        print("\n")
        print_separator()
        print("              COST ANALYSIS")
        print_separator()

        show_value("Input Cost", f"${economics['input_cost']:.6f}")
        show_value("Output Cost", f"${economics['output_cost']:.6f}")
        show_value("Total Cost / Request", f"${economics['total_cost']:.6f}")

        # CONTEXT ANALYSIS
        print("\n")
        print_separator()
        print("          CONTEXT WINDOW ANALYSIS")
        print_separator()

        if economics["context_limit"] is not None:
            show_value("Context Window", f"{economics['context_limit']:,} tokens")
            show_value("Context Used", f"{economics['context_percentage']:.4f}%")

        show_value("Context Status", economics["context_status"])

        if economics["output_limit"] is not None:
            show_value("Maximum Output", f"{economics['output_limit']:,} tokens")

        show_value("Output Status", economics["output_status"])

        # RATE LIMIT ANALYSIS
        print("\n")
        print_separator()
        print("             RATE LIMIT ANALYSIS")
        print_separator()

        show_value("Estimated Daily Requests", f"{economics['daily_requests']:,}")

        if economics["rpm_limit"] is not None:
            show_value("RPM Limit", economics["rpm_limit"])
            show_value("Average RPM Required", f"{economics['estimated_rpm']:.2f}")
            show_value("RPM Status", economics["rpm_status"])

        if economics["tpm_limit"] is not None:
            show_value("TPM Limit", f"{economics['tpm_limit']:,}")
            show_value("Average Input TPM", f"{economics['estimated_tpm']:.2f}")
            show_value("TPM Status", economics["tpm_status"])

        if economics["rpd_limit"] is not None:
            show_value("RPD Limit", f"{economics['rpd_limit']:,}")
            show_value("RPD Status", economics["rpd_status"])

        if economics["max_daily_requests"] is not None:
            show_value(
                "Estimated Daily Capacity",
                f"{economics['max_daily_requests']:,} requests"
            )

        # FEATURE BUDGET
        print("\n")
        print_separator()
        print("             FEATURE BUDGET")
        print_separator()

        show_value("Users", "1,000")
        show_value("Prompts Per User / Day", "10")
        show_value("Daily Requests", f"{economics['daily_requests']:,}")
        show_value("Monthly Requests", f"{economics['monthly_requests']:,}")

        show_value("Estimated Daily Cost", f"${economics['daily_cost']:.4f}")
        show_value("Estimated Monthly Cost", f"${economics['monthly_cost']:.4f}")

        show_value("Monthly Spending Cap", f"${economics['monthly_budget']:.2f}")
        show_value("Remaining Budget", f"${economics['remaining_budget']:.4f}")

        print("\n")
        print_separator()
        print(f"       BUDGET STATUS: {economics['budget_status']}")
        print_separator()

        # FINAL FEASIBILITY CHECK
        failed_limits = [
            economics["rpm_status"],
            economics["tpm_status"],
            economics["rpd_status"],
            economics["context_status"],
            economics["output_status"]
        ]

        if "EXCEEDED" in failed_limits:
            print("\nWARNING: One or more configured limits are exceeded.")

        if economics["rpd_status"] == "EXCEEDED":
            print("Daily request demand exceeds the provider's daily quota.")

        if economics["budget_status"] == "BUDGET EXCEEDED":
            print("Projected monthly spending exceeds the configured budget.")

        print("\n")
        print_separator()
        print("         ANALYSIS COMPLETED SUCCESSFULLY")
        print_separator()

        print("\nNote:")
        print("- Rate estimates assume requests are evenly distributed over 24 hours.")
        print("- Actual peak traffic may exceed RPM or TPM limits.")
        print("- Context usage is measured after generation using actual token counts.")
        print("- Ollama API cost excludes hardware and electricity.")

    except Exception as error:
        print(f"\nAn error occurred: {error}")


if __name__ == "__main__":
    main()
