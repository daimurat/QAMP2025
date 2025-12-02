last_assistant_message = None
for message in reversed(st.session_state.messages):
    if message["role"] == "assistant" and "```" in message["content"]:
        last_assistant_message = message["content"]
        break

if last_assistant_message:
    st.markdown("Executing code...")
    st.info("🚀 Executing cleaned code...")
    execution_output, plot_path = executor.execute_code(last_assistant_message)
    
    st.subheader("Execution Output")
    st.text(execution_output)  # now contains both STDOUT and STDERR
    
    if os.path.exists(plot_path):
        st.success("✅ Plot generated successfully!")
        # Display the plot
        st.image(plot_path, width=700)
    else:
        st.warning("⚠️ No plot was generated")
    
    has_errors = any(error_indicator in execution_output for error_indicator in ["Traceback", "Error:", "Exception:", "TypeError:", "ValueError:", "NameError:", "SyntaxError:", "Error in Class"])

    
    # Check for errors and iterate if needed
    max_iterations = 3  # Maximum number of iterations to prevent infinite loops
    current_iteration = 0
    
    while has_errors and current_iteration < max_iterations:
        current_iteration += 1
        st.error(f"Previous error: {execution_output}")  # Show the actual error message
        st.info(f"🔧 Fixing errors (attempt {current_iteration}/{max_iterations})...")

        # get context on error message
        retrieval_tool = RetrievalTool(vector_store=st.session_state.vector_store)
        context = retrieval_tool.retrieve(execution_output)

        review_message = f"""
        Context:\n{context}\n\nQuestion:

        Previous answer had errors during execution:
        {execution_output}

        Please modify the code to fix those errors. IMPORTANT: Preserve all code blocks exactly as they are, only fix actual errors:
        {last_assistant_message}
        """


        # initialise context to update agent messages
        shared_context = ContextVariables(data =  {
            "user_prompt": "Correct the errors in the code",
            "last_answer": last_assistant_message,
            "feedback": f" Previous answer had errors during execution: {execution_output}",
            "rating": 0,
            "revisions": 0,
        })

        if st.session_state.selected_model in GEMINI_MODELS:
            pattern = AutoPattern(
                initial_agent=refine_agent_gai,
                agents=[refine_agent_gai],
                group_manager_args={"llm_config": initial_config_gai},
                context_variables=shared_context,
            )
        else:
            pattern = AutoPattern(
                initial_agent=refine_agent_final,
                agents=[refine_agent_final],
                group_manager_args={"llm_config": initial_config},
                context_variables=shared_context,
            )
        
        result, context_variables, last_agent = initiate_group_chat(
            pattern=pattern,
            messages=review_message,
            max_rounds=2,
        )

        formatted_answer = result.chat_history[-1]["content"]
        if st.session_state.debug:
            st.session_state.debug_messages.append(("Error Review Feedback", formatted_answer))


        # Execute the corrected code
        st.info("🚀 Executing corrected code...")
        execution_output, plot_path = executor.execute_code(formatted_answer)
        
        st.subheader("Execution Output")
        st.text(execution_output)  # now contains both STDOUT and STDERR
        
        if os.path.exists(plot_path):
            st.success("✅ Plot generated successfully!")
            # Display the plot
            st.image(plot_path, width=700)
        else:
            st.warning("⚠️ No plot was generated")
        
        if st.session_state.debug:
            st.session_state.debug_messages.append(("Execution Output", execution_output))
        
        # Update last_assistant_message with the formatted answer for next iteration
        last_assistant_message = formatted_answer
        has_errors = any(error_indicator in execution_output for error_indicator in ["Traceback", "Error:", "Exception:", "TypeError:", "ValueError:", "NameError:", "SyntaxError:", "Error in Class"])

    if has_errors:
        st.markdown("> ⚠️ **Note**: Some errors could not be fixed after multiple attempts. You can request changes by describing them in the chat.")
        st.markdown(f"> ❌ Last execution message:\n{execution_output}")

        # Display the final code that was successfully executed
        with st.expander("View Failed Code", expanded=False):
            st.markdown(last_assistant_message)
        response = Response(content=f"Execution completed with errors:\n{execution_output}\n\nThe following code was executed:\n```python\n{last_assistant_message}\n")
    else:
        # Check for common error indicators in the output
        if any(error_indicator in execution_output for error_indicator in ["Traceback", "Error:", "Exception:", "TypeError:", "ValueError:", "NameError:", "SyntaxError:"]):
            st.markdown("> ⚠️ **Note**: Code execution completed but with errors. You can request changes by describing them in the chat.")
            st.markdown(f"> ❌ Execution message:\n{execution_output}")
            
                # Display the final code that was successfully executed
            with st.expander("View Failed Code", expanded=False):
                st.markdown(last_assistant_message)
            response = Response(content=f"Execution completed with errors:\n{execution_output}\n\nThe following code was executed:\n```python\n{last_assistant_message}\n")

        else:
            st.markdown(f"> ✅ Code executed successfully. Last execution message:\n{execution_output}")
            
            # Display the final code that was successfully executed
            with st.expander("View Successfully Executed Code", expanded=False):
                st.markdown(last_assistant_message)
                
            # Create a response message that includes the plot path
            response_text = f"Execution completed successfully:\n{execution_output}\n\nThe following code was executed:\n```python\n{last_assistant_message}\n```"
            
            # Add plot path marker for rendering in the conversation
            if os.path.exists(plot_path):
                response_text += f"\n\nPLOT_PATH:{plot_path}\n"
                
            response = Response(content=response_text)
else:
    response = Response(content="No code found to execute in the previous messages.")

return response