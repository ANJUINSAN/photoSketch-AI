import React from 'react'
import './newletters.css'

export default function Newletters() {
  function handleClick(e) {
    
  }
  return (
    <div className='newsletter'>
      <h1>Get Exclusive Offers on your Email</h1>
      <p>Subscribe to our newsletter and stay updated</p>
      <div>
        <input type="email" placeholder='email' />
        <button onClick={()=>{handleClick}} >Subscribe</button>
      </div>
    </div>
  )
}
