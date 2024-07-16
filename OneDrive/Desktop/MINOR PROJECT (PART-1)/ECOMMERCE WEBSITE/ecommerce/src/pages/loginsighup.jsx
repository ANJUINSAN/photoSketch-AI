import React from "react";
import './CSS/loginsignup.css'
import Models from "./models";

export default function Loginsighup() {
 function handleClick(params) {
  return(
    <>
    <Models></Models>
    </>
  )
 }
  return (
    <div className="loginsignup">
      <div className="loginsignup-container">
        <h1>Sign Up</h1>
        <div className="loginsignup-fields">
          <input type="text" placeholder="Your Name" />
          <input type="email" name="" id="" placeholder="Email Address" />
          <input type="password" name="" id="" placeholder="Password" />
        </div>
        <button onClick={()=>handleClick}>Continue</button>
        <p className="loginsignup-login">
          Already have an account? <span>Login Here</span>
        </p>
        <div className="loginsignup-agree">
          <input type="checkbox" name="" id=""  />
          <p>By continuing, I agree to the terms of use & privacy policy.</p>
        </div>
      </div>
    </div>
  );
}
